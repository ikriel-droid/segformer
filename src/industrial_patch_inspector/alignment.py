from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AlignmentConfig
from .structures import AlignmentResult


class CanonicalAligner:
    def __init__(self, config: AlignmentConfig) -> None:
        self.config = config
        self.template_image = self._read_optional_image(config.template_image, grayscale=False)
        self.template_roi = self._read_optional_image(config.template_roi, grayscale=True)
        self.template_gray = (
            cv2.cvtColor(self.template_image, cv2.COLOR_RGB2GRAY)
            if self.template_image is not None
            else None
        )

    @staticmethod
    def _read_optional_image(path: str, grayscale: bool) -> np.ndarray | None:
        if not path:
            return None
        image_path = Path(path)
        if not image_path.exists():
            return None
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(image_path), flag)
        if image is None:
            return None
        if grayscale:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def align(self, image: np.ndarray) -> AlignmentResult:
        method = self.config.method.lower()
        if self.template_image is None or method == "identity":
            roi = self._default_roi(image.shape[:2])
            return AlignmentResult(
                image=image,
                roi_mask=roi,
                transform=np.eye(3, dtype=np.float32),
                score=1.0,
            )

        if method == "ecc":
            return self._align_with_ecc(image)
        if method == "keypoint":
            return self._align_with_keypoints(image)

        roi = self.template_roi if self.template_roi is not None else self._default_roi(image.shape[:2])
        return AlignmentResult(
            image=image,
            roi_mask=roi,
            transform=np.eye(3, dtype=np.float32),
            score=1.0,
        )

    def _align_with_ecc(self, image: np.ndarray) -> AlignmentResult:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.config.ecc_iterations,
            self.config.ecc_eps,
        )
        score = 0.0
        try:
            score, warp = cv2.findTransformECC(
                self.template_gray,
                image_gray,
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
            )
        except cv2.error:
            return AlignmentResult(
                image=image,
                roi_mask=self._default_roi(image.shape[:2]),
                transform=np.eye(3, dtype=np.float32),
                score=0.0,
            )

        warped_image = cv2.warpAffine(
            image,
            warp,
            (self.template_image.shape[1], self.template_image.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        roi_mask = self._warp_roi(warped_image.shape[:2], warp=warp)
        transform = np.eye(3, dtype=np.float32)
        transform[:2, :] = warp
        return AlignmentResult(
            image=warped_image,
            roi_mask=roi_mask,
            transform=transform,
            score=float(max(score, 0.0)),
        )

    def _align_with_keypoints(self, image: np.ndarray) -> AlignmentResult:
        if self.template_gray is None:
            return AlignmentResult(
                image=image,
                roi_mask=self._default_roi(image.shape[:2]),
                transform=np.eye(3, dtype=np.float32),
                score=0.0,
            )

        orb = cv2.ORB_create(nfeatures=self.config.max_keypoints)
        source_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kp_src, des_src = orb.detectAndCompute(source_gray, None)
        kp_tpl, des_tpl = orb.detectAndCompute(self.template_gray, None)
        if des_src is None or des_tpl is None or len(kp_src) < 4 or len(kp_tpl) < 4:
            return AlignmentResult(
                image=image,
                roi_mask=self._default_roi(image.shape[:2]),
                transform=np.eye(3, dtype=np.float32),
                score=0.0,
            )

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = matcher.knnMatch(des_src, des_tpl, k=2)
        good: list[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < 0.75 * second.distance:
                good.append(best)

        if len(good) < 4:
            return AlignmentResult(
                image=image,
                roi_mask=self._default_roi(image.shape[:2]),
                transform=np.eye(3, dtype=np.float32),
                score=0.0,
            )

        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_tpl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is None:
            return AlignmentResult(
                image=image,
                roi_mask=self._default_roi(image.shape[:2]),
                transform=np.eye(3, dtype=np.float32),
                score=0.0,
            )

        warped = cv2.warpPerspective(
            image,
            homography,
            (self.template_image.shape[1], self.template_image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        roi_mask = self._warp_roi(warped.shape[:2], homography=homography)
        inlier_score = float(inlier_mask.mean()) if inlier_mask is not None else 0.0
        return AlignmentResult(
            image=warped,
            roi_mask=roi_mask,
            transform=homography.astype(np.float32),
            score=inlier_score,
        )

    def _warp_roi(
        self,
        target_shape: tuple[int, int],
        warp: np.ndarray | None = None,
        homography: np.ndarray | None = None,
    ) -> np.ndarray:
        roi = self.template_roi
        if roi is None:
            return self._default_roi(target_shape)
        if homography is not None:
            warped = cv2.warpPerspective(
                roi,
                np.linalg.inv(homography),
                (target_shape[1], target_shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        elif warp is not None:
            warped = cv2.warpAffine(
                roi,
                warp,
                (target_shape[1], target_shape[0]),
                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            return self._default_roi(target_shape)
        return (warped > 0).astype(np.uint8) * 255

    @staticmethod
    def _default_roi(shape: tuple[int, int]) -> np.ndarray:
        return np.full(shape, 255, dtype=np.uint8)

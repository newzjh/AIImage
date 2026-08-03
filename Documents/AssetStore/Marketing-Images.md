# Marketing Images

Run `Tools/BuildAssetStoreImages.ps1` from the repository root to rebuild the English-language PNG set in `Documents/AssetStore/MarketingImages/`.

## Upload Set

| Publisher Portal slot | File | Dimensions | Use |
| --- | --- | ---: | --- |
| Cover image | `aexis-cover.png` | 1950 x 1300 | Primary product image. |
| Card image | `aexis-card.png` | 420 x 280 | Store search and browse card. |
| Icon | `aexis-icon.png` | 160 x 160 | Compact product identity. |
| Social image | `aexis-social.png` | 1200 x 630 | Social sharing and external announcement. |
| Screenshot gallery | `showcase-*.png` | 1920 x 1080 | Product-page screenshot gallery. |

## Content and Provenance

- The Cover, Card, Icon, and Social image are product-identification artwork. They use the Aexis name, an engineered GPU motif, and an actual AIImage Main2 screenshot where appropriate.
- Every `showcase-*.png` is assembled from the actual runner artifacts already used by the repository README and package documentation. Their captions identify the runner and preserve output limitations.
- The script writes all text locally to guarantee English spelling and legibility. It does not alter source runner images.
- A GPT Image 2 background-generation attempt was made for this delivery, but the configured API deployment returned `503 model_not_found`. No unavailable or synthetic GPT Image output is represented as a runner result. The current marketing assets are therefore fully reproducible from repository evidence and procedural layout only.

## Visual Review Before Upload

- Inspect each image at 100% scale and at the destination slot size.
- Verify there is no clipped text, watermark, credential, private path, or development-only error.
- Confirm that the actual image inputs are licensed for marketing use.
- Keep the raw-output disclaimer in the GFPGAN, DeepFillV2, and Stable Diffusion showcase captions.
- Re-run the script only after intentional changes to the underlying evidence images or product claims.

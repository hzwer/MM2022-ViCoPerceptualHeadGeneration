# Perceptual Conversational Head Generation with Regularized Driver and Enhanced Renderer
## Introduction
This project is the implement of [Perceptual Conversational Head Generation with Regularized Driver and Enhanced Renderer](https://arxiv.org/abs/2206.12837). We ranked first place in the listening head generation track and second place in the talking head generation track in the official ranking of [MM2022-ViCo Conversational Head Generation Challenge](https://vico.solutions). Our team name is Megvii_goodjuice.

The whole pipeline of challenge can be found on [vico challenge baseline](https://github.com/dc3ea9f/vico_challenge_baseline). We currently provide our major modification of the baseline.

* Image Boundary Inpainting

Change the padding mode of grid_sample in [PIRenderer](https://github.com/RenYurui/PIRender/blob/d75a849978c2eb5f20132b7f0f689c9004d54a00/util/flow_util.py#L56) from "zeros" to "border".

* Fusion

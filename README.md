# Perceptual Conversational Head Generation with Regularized Driver and Enhanced Renderer
## Introduction
This project is the implement of [Perceptual Conversational Head Generation with Regularized Driver and Enhanced Renderer](https://arxiv.org/abs/2206.12837). We ranked first place in the listening head generation track and second place in the talking head generation track in the official ranking of [MM2022-ViCo Conversational Head Generation Challenge](https://vico.solutions). Our team name is Megvii_goodjuice.

The whole pipeline of challenge can be found on [vico challenge baseline](https://github.com/dc3ea9f/vico_challenge_baseline). We currently provide our major modification of the baseline.

* Image Boundary Inpainting

Change the padding mode of grid_sample in [PIRenderer](https://github.com/RenYurui/PIRender/blob/d75a849978c2eb5f20132b7f0f689c9004d54a00/util/flow_util.py#L56) from "zeros" to "border".

* Fusion
Use [U2Net](https://github.com/xuebinqin/U-2-Net) to segment the backgrounds and fuse them to remain the background unchanged.

* Usage
Clone [vico challenge baseline](https://github.com/dc3ea9f/vico_challenge_baseline), and replace `vico_challenge_baseline/vico/networks/audio_to_face.py`, `vico_challenge_baseline/vico/networks/speaker_generator.py`, `vico_challenge_baseline/PIRender/inference.py`, and move `u2net.py` to `vico_challenge_baseline/PIRender/u2net.py`.

You should also download U2Net weights `u2net_human_seg.pth` from [Google Drive](https://drive.google.com/file/d/1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P/view?usp=sharing) and save in `vico_challenge_baseline/PIRender/u2net_human_seg.pth`

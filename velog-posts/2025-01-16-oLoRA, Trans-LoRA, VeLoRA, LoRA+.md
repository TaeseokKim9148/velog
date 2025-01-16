---
title: "oLoRA, Trans-LoRA, VeLoRA, LoRA+"
date: Thu, 16 Jan 2025 10:08:58 GMT
categories: Velog
link: https://velog.io/@kim_taixi/oLoRA
---

<p><strong>Parameter-Efficient Fine-Tuning (PEFT)</strong>은 모델의 모든 파라미터를 조정하지 않고, 특정 파라미터 집합만을 업데이트하여 모델을 튜닝하는 방법</p>
<h2 id="olora">oLoRA</h2>
<ul>
<li>OLoRA는 QR 분해를 통해 직교 행렬(orthonormal matrix)을 초기화하여 모델 학습의 수렴 속도를 크게 향상 </li>
<li>OLoRA는 LoRA의 효율성(학습 가능한 매개변수 수 및 GPU 메모리 사용량)을 유지하면서도, LLM 학습의 수렴을 가속화하고 성능을 향상</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/b1aa3916-8f06-4ffc-a101-88681adc76d5/image.png" /></p>
<h2 id="trans-lora">Trans-LoRA</h2>
<ul>
<li>Trans-LoRA. LoRA 모듈을 기본 모델 간에 손실 없이, 거의 데이터 없이 전이할 수 있는 새로운 방법</li>
</ul>
<p><strong>프로세스 단계</strong></p>
<p>원본 모델(Source Model)</p>
<ul>
<li>LoRA 훈련: 원본 모델에서 초기 LoRA를 훈련</li>
<li>합성 데이터 생성: 원본 모델을 활용해 합성 데이터를 생성, 이를 기반으로 이후 단계를 수행</li>
<li>LoRA 판별기(Discriminator) 훈련: 생성된 합성 데이터를 사용하여 LoRA 판별기를 훈련, 판별기는 데이터의 품질을 평가하고, 전이에 적합한 데이터를 필터링하는 역할을 수행</li>
</ul>
<p>대상 모델(Target Model)</p>
<ul>
<li>합성 데이터 생성 및 필터링: 대상 모델에서 데이터를 생성하고, 판별기를 통해 필터링하여 전이 품질을 보장</li>
<li>대상 LoRA 훈련: 원본 LoRA를 <strong>교사 모델(teacher)</strong>로 사용하여 대상 모델의 LoRA를 학습</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/af233938-4c23-4c73-a37a-36685e8ffe94/image.png" /></p>
<h2 id="velora">VeLoRA</h2>
<p>과정</p>
<ul>
<li>입력 토큰을 <strong>서브 토큰(sub-tokens)</strong>으로 분할</li>
<li>서브 토큰을 고정된 1차원(rank-1) 하위 공간으로 투영하여 압축</li>
<li>역전파(backward pass) 시 압축된 데이터를 복원해 그래디언트 계산에 사용</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/d8217183-06da-496e-9617-145ce244a7a4/image.png" /></p>
<h2 id="lora">LoRA+</h2>
<ul>
<li>어댑터 행렬𝐴와𝐵에 고정된 비율의 서로 다른 학습률을 적용</li>
<li>학습률 차이를 도입하는 단순한 변화로, LoRA의 한계를 극복하며 대규모 모델에서 더 빠르고 효율적인 학습을 가능</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/900064f1-7f4a-41f8-9517-14888c1806cd/image.png" /></p>
<h2 id="참고자료">참고자료</h2>

---
title: "LoRA: Low-Rank Adaptation of Large Language Models[논문리뷰]"
date: Tue, 26 Nov 2024 07:07:54 GMT
categories: Velog
link: https://velog.io/@kim_taixi/LoRA-Low-Rank-Adaptation-of-Large-Language-Models%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0
---

<h3 id="논문-제목lora-low-rank-adaptation-of-large-language-models">논문 제목:LoRA: Low-Rank Adaptation of Large Language Models</h3>
<h3 id="저자--edward-hu-yelong-shen-microsoft-corporation그밖에-팀원들">저자 : Edward Hu, Yelong Shen ,Microsoft Corporation그밖에 팀원들</h3>
<blockquote>
<h2 id="abstract">Abstract</h2>
</blockquote>
<p>대형 언어 모델(LLM)의 미세 조정에는 막대한 GPU 메모리가 필요하며, 이로 인해 더 큰 모델을 사용하는 데 제약이 생깁니다. 로우랭크 적응(Low-Rank Adaptation) 기법의 양자화된 버전인 QLoRA는 이러한 문제를 상당 부분 완화하지만, 효율적인 LoRA 랭크(rank)를 찾는 것은 여전히 어려운 과제이며,QLoRA는 미리 정의된 랭크에서 학습되기 때문에, 더 낮은 랭크로 재구성하려면 추가적인 미세 조정 단계를 요구함 </p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/82877277-b790-4eb4-a66e-3b05ec278399/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/5dfbeff3-ad49-4c07-92ce-bac0bbec3010/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/b8e72fab-92ea-494e-bd1f-097fb7744ca3/image.png" /></p>
<blockquote>
<h2 id="introduction">INTRODUCTION</h2>
</blockquote>
<p>자연어 처리의 많은 애플리케이션에서는 하나의 대규모 사전 학습 언어 모델을 여러 다운스트림 애플리케이션에 맞게 적응시키는 것이 필요합니다. 이러한 적응은 보통 fine-tuning을 통해 이루어지며, 이는 사전 학습된 모델의 모든 파라미터를 업데이트합니다. 그러나 fine-tuning의 주요 단점은 새로운 모델이 원래 모델과 동일한 수의 파라미터를 포함하게 된다는 점입니다. 많은 파라미터를 학습시는것은  시간도 오래 걸릴 뿐더러 비효율이 발생한다.  이를 해결하기 위해 많은 사람들이  일부 파라미터만 적응시키거나 새로운 작업을 위해 외부 모듈을 학습시키는 방법을 모색했습니다. 이렇게 하면 각 작업에 대해 사전 학습된 모델 외에 작업별 파라미터만 저장하고 로드하면 되므로, 배포 시 운영 효율성이 크게 향상됩니다. 하지만 기존 기법들은 모델 깊이를 확장하여 추론 지연 시간을 유발하거나,모델의 사용 가능한 시퀀스 길이를 줄이는 문제를 발생시킵니다. 더 중요한 것은, 이러한 방법들이 종종 파인튜닝 기준치에 미치지 못해 효율성과 모델 품질 사이의 트레이드오프가 발생한다는 점입니다. 그래서 저자들은  <strong>저차원 적응(LoRA, Low-Rank Adaptation)</strong> 접근 방식을 제안합니다. LoRA는 사전 학습된 가중치를 고정한 채, 모델 적응 동안 변화하는 밀집 계층의 랭크 분해 행렬을 최적화하여, 밀집 계층의 일부를 간접적으로 학습할 수 있게 합니다.</p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/4dabceb7-1f32-4c28-88d0-862a64bb2926/image.png" /></p>
<p>LoRA의 장점 : </p>
<ul>
<li><strong>사전 학습된 모델을 공유하여 다양한 작업에 맞춘 작은 LoRA 모듈을 만들 수 있습니다.</strong></li>
<li><strong>LoRA는 훈련 효율을 높이고, 적응형 최적화 기법을 사용할 때 최대 3배까지 하드웨어 진입 장벽을 낮춥니다.</strong></li>
<li><strong>LoRA의 간단한 선형 설계는 배포 시 학습 가능한 행렬을 고정된 가중치와 합칠 수 있어, 완전히 파인튜닝된 모델과 비교할 때 추론 지연이 발생하지 않습니다.</strong></li>
<li><strong>LoRA는 기존의 여러 방법과 독립적으로 적용될 수 있으며, 다양한 방법과 결합이 가능합니다.</strong></li>
</ul>
<p>용어</p>
<ul>
<li><strong>dmodel</strong>: Transformer 레이어의 입력 및 출력 차원 크기를 나타냄</li>
<li><strong>Wq, Wk, Wv, Wo</strong>: 자기-어텐션 모듈의 쿼리, 키, 값, 출력 프로젝션 행렬을 나타냄</li>
<li><strong>W 또는 W0</strong>: 사전 학습된 가중치 행렬을 나타내며, W는 적응 과정에서 누적된 그래디언트 업데이트를 의미함</li>
<li><strong>r</strong>: LoRA 모듈의 랭크(차원 순위)를 나타냄</li>
</ul>
<blockquote>
<h2 id="problem-statement">PROBLEM STATEMENT</h2>
</blockquote>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/1ccba7d0-a276-4617-87ff-230011bb80dd/image.png" /></p>
<ul>
<li>기존의 finetuning 방식인 모델의 모든 파라미터를 업데이트하는 최대우도법 목적식</li>
<li>각 downstream task마다 다른 LoRA layer를 사용해 효율적으로 파라미터를 업데이트 할 수 있도록 하는 것</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/530c4a73-d1d6-4929-bd8c-393605471c6d/image.png" /></p>
<ul>
<li>LoRA를 사용하여 더 작은 rank representation을 제안하여 메모리 및 연산 효율성을 극대화</li>
</ul>
<blockquote>
<h2 id="arent-existing-solutions-good-enough">AREN’T EXISTING SOLUTIONS GOOD ENOUGH?</h2>
</blockquote>
<p>Transfer learning이 시작된 이래 수십 개의 연구들이 모델 adaptation을 보다 파라미터 및 계산 효율적으로 만들기 위해 노력했다. 예를 들어 언어 모델링을 사용하면 효율적인 adaptation과 관련하여 두 가지 주요 전략이 있다.</p>
<ol>
<li>Adapter layer 추가</li>
<li>입력 레이어 activation의 일부 형식 최적화</li>
</ol>
<ul>
<li>문제는 현재의 해결책이 충분하지 않다는 점</li>
<li>효율적인 Adaptation을 위한 두 가지 주요 전략</li>
</ul>
<ol>
<li><strong>Adapter layer는 inference latency를 유발</strong></li>
</ol>
<ul>
<li>다양한 adapter variation이 존재</li>
<li>원래 디자인은 각 Transformer 블록에 두 개의 adapter layer가 있음</li>
<li>최근 디자인은 블록당 하나의 layer와 추가 LayerNorm이 있음</li>
<li>Adapter layer를 사용할 때 latency가 증가</li>
</ul>
<ol>
<li><strong>Prompt 최적화의 어려움</strong></li>
</ol>
<ul>
<li>Prompt tuning은 최적화가 어려움</li>
<li>학습 가능한 parameter 수가 변할 때 성능이 비선형적으로 변화함
<img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/9d5b9f7c-b25d-4905-b4e9-3826c219b7fa/image.png" /></li>
</ul>
<blockquote>
<h2 id="our-method">OUR METHOD</h2>
</blockquote>
<blockquote>
<blockquote>
<h3 id="low-rank-parametrized-update-matrices">LOW-RANK-PARAMETRIZED UPDATE MATRICES</h3>
</blockquote>
</blockquote>
<ul>
<li>특정 task에 adaptation 할 때, 연구에 영감을 받아 parameter 업데이트를 낮은 rank로 제약함</li>
<li>Pre-trained weight <em>W</em>0를 <em>W</em>0+Δ<em>W</em>로 업데이트하는데, Δ<em>W</em>=<em>BA</em>로 분해</li>
<li>Δ<em>W</em>=<em>B</em>⋅<em>A</em>이며, <em>B</em>와 <em>A</em>는 학습 가능한 parameter</li>
</ul>
<ul>
<li><p>위 수식은 pre-trained 모델의 weight <em>W</em>0에 low-rank matix <em>B</em>와 <em>A</em>로 표현된 변화량 Δ<em>W</em>를 더하여 입력 <em>x</em>에 대한 출력을 계산
<img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/70aa4c36-4f7c-4c4b-8478-8d31cd74abdb/image.png" /></p>
<blockquote>
<blockquote>
<h3 id="applying-lora-to-transformer">APPLYING LoRA TO TRANSFORMER</h3>
</blockquote>
</blockquote>
</li>
<li><p>Transformer 아키텍처에서 LoRA를 사용하여 학습 가능한 parameter 수를 줄임</p>
</li>
<li><p>연구는 attention weights에만 적용</p>
</li>
<li><p>메모리와 저장 공간 사용이 크게 줄어듦</p>
</li>
</ul>
<blockquote>
<h2 id="empirical-experiments">Empirical Experiments</h2>
</blockquote>
<h3 id="1-baseline"><strong>1. Baseline</strong></h3>
<ul>
<li><strong>FT</strong>: Fine-Tuning</li>
<li><strong>FTTop2</strong>: 마지막 두 레이어만 튜닝</li>
<li><strong>BitFit</strong></li>
<li><strong>AdapH</strong>: 오리지널 adapter tuning</li>
<li><strong>AdapL</strong>: MLP 모듈 뒤와 LayerNorm 뒤에만 adapter layer 적용</li>
<li><strong>AdapP</strong>: <a href="https://arxiv.org/abs/2005.00247">AdapterFusion</a> (Adap과 유사)</li>
<li><strong>AdapD</strong>: <a href="https://arxiv.org/abs/2010.11918">AdapterDrop</a> (몇몇 adapter layer를 drop)</li>
</ul>
<h3 id="2-result"><strong>2. Result</strong></h3>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/35ad5cb6-364c-4ace-861f-1236deaba846/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/8dc46ecc-d289-4ca0-9292-084b5bf7a743/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/7e078ca6-99b7-468b-83af-b46f9ad9ed1d/image.png" />
<img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/1b2e8c64-5d1e-489a-bdfd-6accf3b54480/image.png" /></p>
<blockquote>
<h2 id="conclusion-and-future-work">CONCLUSION AND FUTURE WORK</h2>
</blockquote>
<p>대한 언어 모델의 파인튜닝은 필요한 하드웨어와, 서로 다른 작업에 대해 독립적인 인스턴스를 호스팅하는 데 필요한 저장 및 전환 비용 측면에서 지나치게 비쌉니다. 저자들은 LoRA라는 효율적인 적응 전략을 제안하며, 이는 모델 품질을 유지하면서도 추론 지연 시간을 초래하지 않고 입력 시퀀스 길이를 줄이지 않습니다. 특히, 대부분의 모델 파라미터를 공유함으로써 서비스로 배포 시 빠른 작업 전환을 가능하게 합니다. 우리는 Transformer 언어 모델에 집중했지만, 제안된 원칙은 밀집 계층이 있는 모든 신경망에 일반적으로 적용할 수 있습니다.</p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/c5573e44-9856-420f-bb1c-027da8c5c745/image.png" /></p>
<h2 id="참고-자료">참고 자료</h2>
<p><a href="https://kyujinpy.tistory.com/83">https://kyujinpy.tistory.com/83</a></p>
<p><a href="https://velog.io/@kameleon43/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-LORA-LOW-RANK-ADAPTATION-OF-LARGE-LANGUAGE-MODELS">https://velog.io/@kameleon43/논문리뷰-LORA-LOW-RANK-ADAPTATION-OF-LARGE-LANGUAGE-MODELS</a></p>
<p><a href="https://jeongwooyeol0106.tistory.com/106">https://jeongwooyeol0106.tistory.com/106</a></p>
<p><a href="https://velog.io/@bluein/paper-22">https://velog.io/@bluein/paper-22</a></p>
<p><a href="https://taeyuplab.tistory.com/12">https://taeyuplab.tistory.com/12</a></p>
<p><a href="https://velog.io/@quasar529/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-LoRA-Low-Rank-Adaptation-of-Large-Language-Models">https://velog.io/@quasar529/논문리뷰-LoRA-Low-Rank-Adaptation-of-Large-Language-Models</a></p>
<p><a href="https://asidefine.tistory.com/309">https://asidefine.tistory.com/309</a></p>

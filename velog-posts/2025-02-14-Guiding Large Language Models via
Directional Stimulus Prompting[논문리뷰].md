---
title: "Guiding Large Language Models via
Directional Stimulus Prompting[논문리뷰]"
date: Fri, 14 Feb 2025 01:28:22 GMT
categories: Velog
link: https://velog.io/@kim_taixi/Guiding-Large-Language-Models-viaDirectional-Stimulus-Prompting%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0
---

<h3 id="논문-제목guiding-large-language-models-via-directional-stimulus-prompting">논문 제목:Guiding Large Language Models via Directional Stimulus Prompting</h3>
<h3 id="저자--zekun-li-baolin-peng-pengcheng-he-michel-galley-jianfeng-gao-xifeng-yan">저자 : Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, Xifeng Yan</h3>
<h3 id="요약">요약</h3>
<ul>
<li>Directional Stimulus Prompting은 대규모 언어 모델(LLMs)을 직접 조정하지 않고, 작은 정책 모델T5등을 통해 각 입력에 맞는 보조 프롬프트를 생성해 LLM을 원하는 방향으로 유도하는 방법 
이를 통해 LLM의 성능을 개선하며, 정책 모델은 지도 학습이나 강화 학습으로 최적화함</li>
</ul>
<h3 id="사용하는곳">사용하는곳</h3>
<ul>
<li>작고 조정 가능한 언어모델을 사용하여 원하는 LLM의 응답을 원하는 결과 로 유도하는 힌트나 단서 제공</li>
<li>기존 미세 조정 방식보다 더 큰 제어력을 제공하며, 모델의 응답을 안해하면서도 모델의 일반적인 능력을 유지</li>
</ul>
<h3 id="특징">특징</h3>
<ol>
<li>가이드 힌트 제공</li>
<li>정책 언어 모델 훈련</li>
<li>정확도와 사용자 선호도 향상</li>
</ol>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/1ab13c70-377a-4f30-8730-8098adb1dab8/image.png" /></p>
<h3 id="작동방식">작동방식</h3>
<ol>
<li>힌트 생성</li>
<li>LLM 출력유도</li>
<li>강화 학습을 통해 최적화 </li>
</ol>
<p><strong>정리</strong></p>
<p>작은 규모의 정책 모델(예: T5)**을 활용하여 각 입력에 맞는 Directional Stimulus를 자동 생성합니다.</p>
<p>지도 학습(SFT) – 소량의 라벨 데이터로 초기 Directional Stimulus 생성 학습
강화 학습(RL) – LLM의 성능(ROUGE 점수, 사용자 선호도 등)을 기준으로 최적화</p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/f2b15866-228e-4c74-ae5a-8eeec01740ff/image.png" /></p>
<h2 id="참고자료">참고자료</h2>
<p><a href="https://slashpage.com/haebom/k5r398nmnnx8emvwje7y?lang=ko">https://slashpage.com/haebom/k5r398nmnnx8emvwje7y?lang=ko</a>
<a href="https://brunch.co.kr/@aichaemun/109">https://brunch.co.kr/@aichaemun/109</a></p>

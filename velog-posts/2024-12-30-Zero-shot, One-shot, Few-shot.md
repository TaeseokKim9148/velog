---
title: "Zero-shot, One-shot, Few-shot"
date: Mon, 30 Dec 2024 11:52:07 GMT
categories: Velog
link: https://velog.io/@kim_taixi/Zero-shot-One-shot-Few-shot
---

<ul>
<li><strong>Zero-shot (ZSL)</strong><ul>
<li>모델이 학습 과정에서 본 적 없는 새로운 데이터를 인식할 수 있도록 하는 학습 방법</li>
<li>모델이 클래스 간의 관계나 속성을 통해 일반화하는 능력 활용</li>
<li>대규모 모델 사이즈로 대규모의 다양한 데이텃으로 학습한 경우 성능이 잘나옴</li>
<li>training과 inference, 두 개의 stage로 이루어짐</li>
</ul>
</li>
<li><strong>One-shot (OSL)</strong><ul>
<li>각 클래스에 대해 단 하나의 데이터만 제공될 때 모델이 그 클래스를 인식할 수 있도록 하는 학습 방법</li>
<li>유사도 학습이나 메타 학습 등의 기법을 활용</li>
<li>학습 데이터가 매우 제한적일 때 유용</li>
</ul>
</li>
<li><strong>Few-shot (FSL)</strong><ul>
<li>극소량의 데이터만을 이용하여 새로운 작업이나 클래스를 빠르게 학습하도록 설계된 알고리즘</li>
<li>메타러닝이나 학습 전략의 최적화 등을 통해 적은 데이터로도 효과적인 일반화 능력을 갖춤</li>
</ul>
</li>
</ul>
<ol>
<li>전이 학습 : 사전학습 모델을 중심으로 학습하고 소량의 데이터로 재학습. </li>
<li>메타 학습 : 여러개의 작업을 동시에 하고 각 작업 간의 차이도 같이 학습하는것을 말함.  </li>
</ol>
<h2 id="참고자료">참고자료</h2>
<p><a href="https://madang-ai.tistory.com/2">https://madang-ai.tistory.com/2</a></p>
<p><a href="https://onestoria.tistory.com/57">https://onestoria.tistory.com/57</a></p>
<p><a href="https://rimiyeyo.tistory.com/entry/%EB%8B%A4%EC%96%91%ED%95%9C-%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81Prompt-Engineering%EC%97%90-%EB%8C%80%ED%95%B4-%EC%82%B4%ED%8E%B4%EB%B3%B4%EC%9E%901-Zero-shot-One-shot-Few-shot-CoT">https://rimiyeyo.tistory.com/entry/%EB%8B%A4%EC%96%91%ED%95%9C-%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81Prompt-Engineering%EC%97%90-%EB%8C%80%ED%95%B4-%EC%82%B4%ED%8E%B4%EB%B3%B4%EC%9E%901-Zero-shot-One-shot-Few-shot-CoT</a></p>
<p><a href="https://dodonam.tistory.com/452#google_vignette">https://dodonam.tistory.com/452#google_vignette</a></p>

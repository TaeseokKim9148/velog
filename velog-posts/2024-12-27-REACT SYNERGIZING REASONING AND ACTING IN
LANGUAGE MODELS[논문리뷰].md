---
title: "REACT: SYNERGIZING REASONING AND ACTING IN
LANGUAGE MODELS[논문리뷰]"
date: Fri, 27 Dec 2024 04:00:12 GMT
categories: Velog
link: https://velog.io/@kim_taixi/REACT-SYNERGIZING-REASONING-AND-ACTING-INLANGUAGE-MODELS%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0
---

<h3 id="논문-제목react-synergizing-reasoning-and-acting-in-language-models">논문 제목:REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS</h3>
<h3 id="저자--shunyu-yao-jeffrey-zhao-dian-yu-nan-du-izhak-shafran-karthik-narasimhan-yuan-cao">저자 : Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao</h3>
<blockquote>
<h2 id="abstract">Abstract</h2>
</blockquote>
<p>ReAct = Reason + Act
           = 추론 + 실행</p>
<p>추론 과정 : 모델이 행동 계획을 유도, 추적, 업데이트하고 예외를 처리하는 데 도움을 줌
행동 과정 : 모델이 외부 지식 베이스나 환경(예: API)에서 추가 정보를 수집하고 이를 활용</p>
<h4 id="특징">특징</h4>
<p>1.추론을 통해 행동(reason to act): 고수준 계획을 생성, 유지, 조정할 수 있음
2.행동을 통해 추론(act to reason): 외부 환경(예: Wikipedia)과 상호작용하여 추가 정보를 추론에 통합할 수 있음</p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/9b92a27e-36d9-4aea-b93b-d35e56fa97a2/image.png" /></p>
<blockquote>
<h2 id="introduction">Introduction</h2>
</blockquote>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/e3432007-18ff-468d-859e-473c331123a0/image.png" /></p>
<h4 id="reason-only-모델은-자체-정보가-부족할-경우-hallucination에-의한-부정확한-정보를-출력-act-only-모델은-추론-능력-부족으로-외부-정보를-기반으로도-최종-답에-이르지-못하고-엉뚱한-대답을-하는-한계가-있다-실제로-action은-react와-똑같이-함반면-react는-해석가능하고-사실에-기반한-trajectory로-task를-해결함">Reason-only 모델은 자체 정보가 부족할 경우 Hallucination에 의한 부정확한 정보를 출력, Act-only 모델은 추론 능력 부족으로 외부 정보를 기반으로도 최종 답에 이르지 못하고 엉뚱한 대답을 하는 한계가 있다. (실제로 Action은 ReAct와 똑같이 함)반면 ReAct는 해석가능하고 사실에 기반한 Trajectory로 task를 해결함</h4>
<blockquote>
<h2 id="react-synergizing-reasoning--acting">REACT: SYNERGIZING REASONING + ACTING</h2>
</blockquote>
<h3 id="데이터셋">데이터셋</h3>
<p>HotPotQA: 두 개 이상의 Wikipedia 문서를 바탕으로 추론해야 하는 멀티홉 질문 응답 벤치마크.
FEVER: 주어진 주장(Claim)이 Wikipedia 문서에 의해 지지(SUPPORTS), 반박(REFUTES) 또는 <strong>불충분한 정보(NOT ENOUGH INFO)</strong>로 분류되는 사실 검증 벤치마크.</p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/17797a2e-edc9-4c9f-9250-d1220908184b/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/e89bc20d-8be7-42a9-8f58-a59df398e64f/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/874f8c9c-a626-4999-b5e2-a15dcb02c52e/image.png" /></p>
<p><strong>결론</strong> : ReAct는 추론과 행동의 결합을 통해 CoT보다 더 높은 성공 비율을 보이며, 외부 정보와 상호작용할 수 있는 능력이 주요 강점으로 작용,실패 사례는 ReAct와 CoT 모두 복잡한 질문이나 다중 단계 추론에서 주로 발생</p>
<h2 id="참고자료">참고자료</h2>
<p><a href="https://arize.com/blog/keys-to-understanding-react/">https://arize.com/blog/keys-to-understanding-react/</a>
<a href="https://introduce-ai.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-ReAct-SYNERGIZING-REASONING-AND-ACTING-IN-LANGUAGE-MODELS">https://introduce-ai.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-ReAct-SYNERGIZING-REASONING-AND-ACTING-IN-LANGUAGE-MODELS</a>
<a href="https://www.youtube.com/watch?v=QX-p-vsDoiQ">https://www.youtube.com/watch?v=QX-p-vsDoiQ</a></p>

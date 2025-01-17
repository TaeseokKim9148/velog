---
title: "LtM(LEAST-TO-MOST PROMPTING ENABLES COMPLEX
REASONING IN LARGE LANGUAGE MODELS)"
date: Fri, 17 Jan 2025 06:03:50 GMT
categories: Velog
link: https://velog.io/@kim_taixi/LtMLEAST-TO-MOST-PROMPTING-ENABLES-COMPLEXREASONING-IN-LARGE-LANGUAGE-MODELS
---

<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/4203667c-25fa-4bab-85b5-45f5da695af6/image.png" /></p>
<ul>
<li>쉬운 문제에서 어려운 문제로 일반화&quot;를 가능</li>
</ul>
<p>과정</p>
<p><strong>Decomposition (문제 분해)</strong></p>
<p>과정: </p>
<ul>
<li><p>주어진 문제를 여러 개의 <strong>하위 문제(subproblems)</strong>로 쪼개는 방법에 대한 예시를 프롬프트로 제시</p>
</li>
<li><p>모델은 이러한 예시를 참고하여 복잡한 문제를 작고 해결 가능한 하위 문제들로 분해</p>
</li>
</ul>
<p><strong>Subproblem Solving (하위 문제 해결)</strong></p>
<p>단계:
하위 문제 풀이 예시 제공:
각 하위 문제가 어떻게 해결되는지 보여주는 예시를 프롬프트로 제시</p>
<p>리스트 사용:
비어 있는 리스트를 제공하며, 해결된 하위 문제의 답과 솔루션을 순차적으로 저장</p>
<p>최종 질문 제시:
하위 문제의 답을 종합하여 최종 질문에 답하도록 구성</p>
<p>순차적 풀이:
하위 문제를 하나씩 해결하며, 이전 답변을 다음 하위 문제의 입력으로 사용.
이 과정을 반복해 최종 결과를 도출</p>

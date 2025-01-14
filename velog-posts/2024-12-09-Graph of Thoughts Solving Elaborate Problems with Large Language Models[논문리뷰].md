---
title: "Graph of Thoughts: Solving Elaborate Problems with Large Language Models[논문리뷰]"
date: Mon, 09 Dec 2024 15:06:22 GMT
categories: Velog
link: https://velog.io/@kim_taixi/Tree-of-Thoughts-Deliberate-Problem-Solvingwith-Large-Language-Models%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-qoln2les
---

<h3 id="논문-제목graph-of-thoughts-solving-elaborate-problems-with-large-language-models">논문 제목:Graph of Thoughts: Solving Elaborate Problems with Large Language Models</h3>
<h3 id="저자--shunyu-yao-dian-yu-jeffrey-zhao-izhak-shafran-thomas-l-griffiths-yuan-cao-karthik-narasimhan">저자 : Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan</h3>
<blockquote>
<h2 id="abstract">Abstract</h2>
</blockquote>
<ul>
<li>아이디어 및 주요 장점은 LLM이 생성하는 정보를 임의의 그래프 형태로 모델링하는 것</li>
<li>GoT는 새로운 생각 변환(Thought Transformation)을 추가할 수 있도록 확장 가능하게 설계되어, 새로운 프롬프팅 기법을 선도</li>
</ul>
<h4 id="특징">특징</h4>
<p>1) 임의의 생각 결합: 서로 다른 LLM의 생각을 결합하여 시너지 효과를 낼 수 있음.
2)생각 네트워크의 본질 추출: 전체적인 생각의 네트워크에서 핵심 정보를 압축하고 추출할 수 있음
3)피드백 루프를 통한 개선: 피드백 메커니즘을 사용해 생각을 지속적으로 향상시킬 수 있음</p>
<blockquote>
<h2 id="introduction">Introduction</h2>
</blockquote>
<ul>
<li><p>GoT는 생각(LLM이 생성하는 정보)을 <strong>그래프의 꼭짓점(Vertex)</strong>으로, 생각 간의 의존 관계를 <strong>에지(Edge)</strong>로 모델링하여 더 유연하고 강력한 추론 패턴</p>
</li>
<li><p><strong>Graph of Thoughts (GoT)</strong>는 프롬프팅 전략 평가를 위해 <strong>생각의 부피(Volume of a Thought)</strong>라는 새로운 지표를 제안</p>
</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/592a269b-28e9-40c8-804b-bb8f02632ab8/image.png" /></p>
<blockquote>
<h2 id="the-got-framework">The GoT Framework</h2>
</blockquote>
<h4 id="기법">기법</h4>
<ul>
<li>Aggregation: 여러 생각을 결합해 새로운 생각을 생성.</li>
<li>Refining: 반복 연결을 통해 생각을 개선.</li>
<li>Generation: 기존 생각을 기반으로 새로운 생각들을 생성.</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/6bc1cf43-e555-4095-a398-c5a35d4c4e30/image.png" /></p>
<ol>
<li><p>아키텍처 개요(파란색)
GoT는 모듈형 시스템으로, 다음 주요 구성 요소로 이루어집니다:
Prompter: LLM에 보낼 프롬프트 생성
Parser: LLM 출력에서 정보 추출
Scoring &amp; Validation: 생각의 정확성 검증 및 점수 부여
Controller: 전체 추론 프로세스 조정 및 다음 단계 결정
Graph of Operations (GoO): 작업 순서와 의존 관계를 정의하는 정적 구조
Graph Reasoning State (GRS): 진행 중인 추론 상태를 저장하는 동적 구조</p>
</li>
<li><p>확장 가능한 API(초록색)
Controller API: 생각 생성, 결합, 반복 제어 (예: Generate, KeepBest).
Prompter API: 프롬프트 생성 및 LLM 전달.
Parser API: LLM 출력 분석 및 상태 업데이트.</p>
</li>
<li><p>예제와 GRS 상태(빨강)
정렬(Sorting) 작업 예시를 통해 GoT의 작동 방식을 설명:
Generate: 시퀀스를 여러 하위 리스트로 나눔.
Aggregate: 하위 리스트를 결합해 정렬된 결과 생성.
Improve: 결과를 반복적으로 검증하고 수정.</p>
</li>
</ol>
<blockquote>
<h2 id="혼합과정">혼합과정</h2>
</blockquote>
<p>1️⃣ CoT + ToT </p>
<p>CoT와 ToT의 통합은 복잡한 문제 해결에 특히 유용할 수 있습니다. CoT를 통해 문제 해결 과정을 단계별로 분해하고, 각 단계에서 ToT를 활용하여 관련 개념을 체계적으로 설명함으로써, AI가 문제에 대한 깊이 있는 이해를 바탕으로 최적의 해결책을 도출할 수 있게 합니다. 예를 들어, 복잡한 공학 문제를 해결할 때 CoT로 문제 해결 절차를 단계화하고, 각 단계에서 필요한 개념과 원리를 ToT로 설명하는 프롬프트를 제공할 수 있습니다.</p>
<p>2️⃣ ToT + GoT</p>
<p>ToT와 GoT 기법의 통합은 개념 간 관계를 명확히 하고 맥락 정보를 풍부하게 제공하는 데 도움이 됩니다. ToT로 개념의 계층 구조를 설명하고, GoT를 통해 개념 간 연결 관계와 속성 정보를 제공함으로써, AI는 주제에 대한 종합적인 이해를 얻을 수 있습니다. 이는 특히 지식 집약적인 분야, 예를 들면 의학이나 법률 분야에서 유용하게 활용될 수 있습니다.</p>
<p>3️⃣ CoT + GoT</p>
<p>CoT와 GoT 기법을 함께 사용하면 복잡한 추론 과정을 지식 그래프와 연계하여 설명할 수 있습니다. CoT로 추론 과정을 단계화하고, 각 단계에서 필요한 지식을 GoT로 제공하는 프롬프트를 통해, AI는 추론의 근거가 되는 지식을 명시적으로 활용할 수 있게 됩니다. 이는 AI의 추론 과정을 투명하게 보여주고, 유저가 AI를 신뢰하도록 할 수 있습니다.</p>
<p>4️⃣ CoT + ToT + GoT</p>
<p>나아가 CoT, ToT, GoT 기법을 모두 통합한 프레임워크를 구축할 수도 있습니다. 예를 들어 복잡한 의사 결정 문제에 직면했을 때, CoT로 의사 결정 단계를 구조화하고, 각 단계에서 ToT와 GoT를 활용하여 고려해야 할 요인과 맥락 정보를 종합적으로 제공할 수 있습니다. 이렇게 통합된 프롬프트는 AI가 문제를 다각도로 분석하고 최선의 솔루션을 도출하는 데 도움을 줄 수 있습니다.</p>
<h2 id="추가자료">추가자료</h2>
<p><a href="https://github.com/spcl/graph-of-thoughts">https://github.com/spcl/graph-of-thoughts</a> (코드)
<a href="https://velog.io/@delee12/LLM-Graph-of-Thoughts-Solving-elaborate-problems-with-large-language-models-AAAI-2024">https://velog.io/@delee12/LLM-Graph-of-Thoughts-Solving-elaborate-problems-with-large-language-models-AAAI-2024</a>
<a href="https://www.youtube.com/watch?v=psVspnBJ9qM">https://www.youtube.com/watch?v=psVspnBJ9qM</a> (동영상 설명)
<a href="https://aiheroes.ai/community/153">https://aiheroes.ai/community/153</a> (혼합과정)</p>

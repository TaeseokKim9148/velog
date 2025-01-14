---
title: "Tree of Thoughts: Deliberate Problem Solving
with Large Language Models[논문리뷰]"
date: Wed, 04 Dec 2024 06:47:21 GMT
categories: Velog
link: https://velog.io/@kim_taixi/Tree-of-Thoughts-Deliberate-Problem-Solvingwith-Large-Language-Models%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0
---

<h3 id="논문-제목tree-of-thoughts-deliberate-problem-solving-with-large-language-models">논문 제목:Tree of Thoughts: Deliberate Problem Solving with Large Language Models</h3>
<h3 id="저자--shunyu-yao-dian-yu-jeffrey-zhao-izhak-shafran-thomas-l-griffiths-yuan-cao-karthik-narasimhan">저자 : Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan</h3>
<blockquote>
<h2 id="abstract">Abstract</h2>
</blockquote>
<p>언어 모델 프롬프팅에서 널리 사용되는 &quot;Chain of Thought (CoT)&quot; 접근 방식을 일반화한 것으로, 문제 해결을 위한 중간 단계 역할을 하는 <strong>&quot;생각(thoughts)&quot;</strong>이라는 일관된 텍스트 단위를 탐구하게 해줌</p>
<h4 id="특징">특징</h4>
<p>1) ToT는 여러 가지 다양한 추론 경로를 고려하고, 스스로 선택을 평가하면서 다음 행동 방향을 결정 필요할 경우, <strong>앞으로 내다보거나 뒤로 돌아가는(backtracking)</strong> 기능을 통해 전체적인 최적의 선택을 가능</p>
<p>2)ToT는 기존의 CoT보다 더 복잡한 문제에서 성능을 크게 향상</p>
<blockquote>
<h2 id="introduction">Introduction</h2>
</blockquote>
<p>언어 모델(GPT, PaLM 등)은 수학적, 상식적, 지식적 추론까지 수행하며 발전왔지만  여전히  기존의 자동회귀 방식(한 번에 하나의 단어를 왼쪽에서 오른쪽으로 생성하는 방식)만으로는 일반 문제 해결에 한계가 있었음.
인간 인지에서 두 가지 결정 모드(fast &amp; automatic과 slow &amp; deliberate)가 있다는 연구에 기반해, 언어 모델에 보다 신중한 계획(deliberative plan) 과정을 추가해야 한다는 아이디어가 제안됨</p>
<p>이러한 계획 과정을 위해, 우리는 뉴웰과 사이먼의 문제 해결 방식을 참고하여 <strong>&quot;Tree of Thoughts, ToT&quot;</strong> 프레임워크를 제안됨. ToT는 문제 해결 과정에서 중간 단계별로 여러 가지 가능한 생각을 탐색하고 평가하는 구조로, 이를 통해 모델이 보다 체계적으로 문제를 해결할 수 있음
<strong>너비 우선 탐색(BFS) 또는 깊이 우선 탐색(DFS) 같은 탐색 알고리즘</strong>을 통해 Tree of Thoughts를 활용합니다.</p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/6ee98bab-a5af-4dde-9d4d-580f0d1e6a7f/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/eee191dc-4b0c-4d1c-bb29-0c5d02190eb2/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/547d8ae7-7c78-4db3-9404-10459d17fdf5/image.png" /></p>
<ul>
<li>CoT에 비해 74%로 좋은 성능을 냈다. </li>
</ul>
<h2 id="추가자료">추가자료</h2>
<p><a href="https://oglee.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Tree-of-Thoughts-Deliberate-Problem-Solving-with-Large-Language-Models">https://oglee.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Tree-of-Thoughts-Deliberate-Problem-Solving-with-Large-Language-Models</a>
<a href="https://lilys.ai/notes/280318">https://lilys.ai/notes/280318</a></p>

---
title: "Scaling Instruction-Finetuned Language Models[논문리뷰]"
date: Mon, 02 Dec 2024 14:06:12 GMT
categories: Velog
link: https://velog.io/@kim_taixi/Scaling-Instruction-Finetuned-Language-Models%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0
---

<h3 id="논문-제목scaling-instruction-finetuned-language-models">논문 제목:Scaling Instruction-Finetuned Language Models</h3>
<h3 id="저자--hyung-won-chung-le-hou-shayne-longpre-barret-zoph-yi-tay-william-fedus-그외-google">저자 : Hyung Won Chung, Le Hou Shayne Longpre, Barret Zoph, Yi Tay William Fedus, 그외 Google</h3>
<blockquote>
<h2 id="abstract">Abstract</h2>
</blockquote>
<p>Instruction Finetuning이란?</p>
<p>모델을 다양한 데이터셋(명령어 형식으로 작성된 데이터)으로 미세 조정하는 방법으로, 이를 통해 모델의 성능과 새로운 작업에 대한 일반화 능력을 향상시킬 수 있음.</p>
<ol>
<li>작업(Task)의 수를 확장</li>
<li>모델 크기를 확장</li>
<li>Chain-of-Thought(생각의 흐름) 데이터를 활용한 미세 조정</li>
</ol>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/4b72bb75-b87f-4df9-bf34-f13c337aad9b/image.png" /></p>
<p>Fine-tuning : 이미 학습된 모델을 가져와서 추가 데이터를 사용하여 다시 학습 시키는 과정. 모델은 기존에 학습한 지식을 활용하면서도, 특정 작업에 더욱 적합한 성능을 발휘한다
Prompt-learning : 모델에게 특정한 프롬프트를 제공하여 학습 시키는 방법. 프롬프트는 모델이 생성해야 하는 결과의 방향성을 제시함으로써, 특정한 작업에 대한 지식을 빠르게 습득할 수 있도록 돕는다
Instruction-tuning : 모델에게 특정한 지시사항을 제공하여 학습. 이 방법은 모델이 그간 마주치지 못했던 작업에도 지시사항을 따라 답 추론이 가능</p>
<blockquote>
<h2 id="introduction">INTRODUCTION</h2>
</blockquote>
<p>Instruction Finetuning(명령어 미세 조정)을 통해 언어 모델의 성능과 일반화 능력을 개선한 내용을 다룸.</p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/60830e42-6556-40a0-b428-01ecedbe3249/image.png" /></p>
<blockquote>
<h2 id="flan-finetuning">Flan Finetuning</h2>
</blockquote>
<p>기존 연구에 따르면, 명령어를 사용한 미세 조정에서 작업 수를 늘리면 보지 못한 작업에 대한 일반화 성능이 향상된다는 것이 입증
<img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/249126d4-92fd-4465-a3c9-24d0f5d35734/image.png" />
다양한 미세 조정 데이터 형식을 사용, 예시는 포함하거나 포함하지 않은 경우, Chain-of-Thought(CoT)는 포함하거나 포함하지 않은 경우를 조합하여 실험</p>
<h4 id="chain-of-thoughtcot-미세-조정-혼합">Chain-of-Thought(CoT) 미세 조정 혼합</h4>
<p>미세 조정 데이터 혼합(추론 관련)은 CoT 주석을 활용하며, 이를 통해 CoT 주석을 사용한 미세 조정이 보지 못한 추론 작업에서 성능을 향상시키는지 탐구</p>
<ul>
<li>Chain-of-Thought Prompting은 언어 모델이 답을 찾아가는 과정에서 &quot;생각을 말로 표현하거나(think aloud)&quot; 단계별 추론 과정(step-by-step reasoning)을 따르도록 유도하는 방식</li>
</ul>
<h3 id="학습">학습</h3>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/c779782e-dd2d-46d4-be58-8d7756468344/image.png" /></p>
<ul>
<li>Finetuning에는 pretraining보다 훨씬 적은 시간을 소모했는데, 예를 들어 Flan-PaLM 학습을 위해 pre-training compute의 0.2%를 소모했다고 함</li>
</ul>
<h3 id="평가">평가</h3>
<ul>
<li>MMLU(수학, 역사, 법, 의학)</li>
<li>BBH(어려운 task들)</li>
<li>TyDiQA(8가지 언어의 QA)</li>
<li>MGSM(다언어 수학 문제)</li>
</ul>
<p>** 전문가의 88%와 비슷하게 정확하게 맞추고 있음 </p>
<blockquote>
<h2 id="scaling-to-540b-parameters-and-18k-tasks">Scaling to 540B parameters and 1.8K tasks</h2>
</blockquote>
<h4 id="instruction-finetuning-효과">Instruction Finetuning 효과</h4>
<p>미세 조정을 통해 모든 모델 크기에서 큰 성능 향상을 달성했으며, 성능 향상 폭은 <strong>9.4%~15.5%</strong>에 이름</p>
<h4 id="작업-수-확장-효과">작업 수 확장 효과</h4>
<p>작업 수를 늘릴수록 성능이 개선되었지만, <strong>282개 작업까지만 대부분의 성능 향상</strong>이 이루어짐.
282개를 초과하는 작업은 다양성이 부족하거나, 모델이 이미 알고 있는 지식을 더 잘 표현하는 데 추가적으로 크게 기여하지 못했습니다.</p>
<h4 id="모델-크기-확장-효과">모델 크기 확장 효과</h4>
<p><strong>모델 크기를 확장하면</strong> 미세 조정 여부와 관계없이 성능이 크게 향상됨.
작은 모델(8B)은 절대적 성능 향상이 더 컸지만, 큰 모델(540B)은 상대적 오류율 감소가 더 컸습니다.</p>
<h4 id="확장-곡선-통찰">확장 곡선 통찰</h4>
<p>모델 크기를 더 확장하면 상당한 성능 향상이 예상되며, 작업 수 확장은 점진적인 성능 개선을 가져올 가능성이 높습니다.</p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/94e329a6-7a06-45b3-a587-ebcb163d87ce/image.png" /></p>
<blockquote>
<h2 id="finetuning-with-chain-of-thought-annotations">Finetuning with chain-of-thought annotations</h2>
</blockquote>
<p>Flan 미세 조정의 목표는 기존 NLP 작업뿐만 아니라 다단계 추론 능력을 포함한 다양한 평가에서 성능을 개선하는 체크포인트를 만드는 것, 현재 섹션에서는 Instruction Finetuning 데이터 혼합물에 CoT 데이터를 포함하는 효과를 연구</p>
<p>🖍️Flan-PaLM은 CoT 데이터와 Self-Consistency를 결합하여 다단계 추론 및 다국어 작업에서 뛰어난 성능을 발휘했으며, 특히 기존 모델들에 비해 큰 개선을 이룸
<img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/418eb952-08c5-479a-bc93-b3627940eec4/image.png" /></p>
<p>** 작업을 늘일수록 성능이 좋아진걸 볼수 있음.</p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/fa30a212-b58c-4832-bb2e-dfda1ba14be3/image.png" /></p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/1da71ed3-566a-427a-8805-b1bd907b9c42/image.png" /></p>
<p>nstruction Finetuning을 통해 Flan-PaLM은 Zero-shot 설정에서도 CoT 추론을 수행할 수 있으며, 이는 미세 조정을 거치지 않은 PaLM과 비교했을 때 큰 성능 차이를 보여줌.</p>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/58aae888-c1b0-49ed-8876-f97a6a3e2479/image.png" /></p>
<ul>
<li>InstructionFine Tunning 과 Chain of thought데이터를 통해 Zero shot 능력을 올림</li>
</ul>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/4d8b3656-54b2-4c1b-9b5b-db0054c81b01/image.png" /></p>
<blockquote>
<h2 id="결론">결론</h2>
</blockquote>
<p><img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/b636a389-e04c-4e98-857e-c930664c17c2/image.png" />
<img alt="" src="https://velog.velcdn.com/images/kim_taixi/post/7e0ae0ad-999f-4d4a-beeb-7eabc4c48282/image.png" /></p>
<h2 id="참고-자료">참고 자료</h2>
<p><a href="https://velog.io/@heomollang/LLaMA-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-1-LLaMA-Open-and-Efficient-Foundation-Language-Models">https://velog.io/@heomollang/LLaMA-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-1-LLaMA-Open-and-Efficient-Foundation-Language-Models</a>
<a href="https://velog.io/@nellcome/Instruction-Tuning%EC%9D%B4%EB%9E%80">https://velog.io/@nellcome/Instruction-Tuning%EC%9D%B4%EB%9E%80</a> (Instruction Tuning이란?)
<a href="https://devocean.sk.com/blog/techBoardDetail.do?ID=165806&amp;boardType=techBlog">https://devocean.sk.com/blog/techBoardDetail.do?ID=165806&amp;boardType=techBlog</a> (Instruction tuning : LLM이 사람 말을 알아 듣는 방법)
<a href="https://zoeoz-ai.tistory.com/5">https://zoeoz-ai.tistory.com/5</a>
<a href="https://velog.io/@sobit/Instruction-Tuning%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81-%ED%8C%8C%EC%9D%B8%ED%8A%9C%EB%8B%9D%EA%B3%BC%EC%9D%98-%EC%B0%A8%EC%9D%B4-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EC%98%88%EC%A0%9C-%EC%BD%94%EB%93%9C">https://velog.io/@sobit/Instruction-Tuning%ED%94%84%EB%A1%AC%ED%94%84%ED%8A%B8-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4%EB%A7%81-%ED%8C%8C%EC%9D%B8%ED%8A%9C%EB%8B%9D%EA%B3%BC%EC%9D%98-%EC%B0%A8%EC%9D%B4-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EC%98%88%EC%A0%9C-%EC%BD%94%EB%93%9C</a>
<a href="https://velog.io/@k106419/Scaling-Instruction-Finetuned-Language-Models-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0">https://velog.io/@k106419/Scaling-Instruction-Finetuned-Language-Models-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0</a> (관련논문리뷰)
<a href="https://rfriend.tistory.com/843">https://rfriend.tistory.com/843</a></p>

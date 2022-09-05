![image](https://user-images.githubusercontent.com/100823210/187594402-4f6a7234-293d-4938-adcb-eb602a50b564.png)
# [ML] 자율주행 센서의 안테나 성능 예측

## 개요
데이콘에서 주최한 대회에 참가하여 머신러닝으로 제품의 성능을 예측하는 프로젝트를 진행했습니다.
라이다 제조 공정에서 발생한 3만 9천개 데이터를 학습 시켰고, 주최측에서 제공한 평가산식(NRMSE) 기준 1.95231 를 얻었습니다.
대회 참가팀 743개 팀 중에 12위로 상위권 성적을 받을 수 있었습니다.

![DACON 대회 결과](https://user-images.githubusercontent.com/100823210/187594733-4becdacd-6606-4745-bf07-8c2c72e41177.png)

## 참여배경

4차 산업이 발전하면서 AI를 이용한 스마트 공장을 도입하는 중소기업이 늘어나고 있습니다. 제품의 불량율을 예측하는 부분에서 많이 활용되고 있으며, 이는 기업의 경쟁력을 높여주고 있습니다. 

이번 프로젝트를 통해 머신러닝으로 제조공정의 데이터를 활용하여 불량률을 예측해보려고 합니다.

> [YTN - 제조 혁신의 아이콘 '스마트 공장'...중소·중견 기업으로 확대](https://n.news.naver.com/mnews/article/052/0001756416?sid=102)

> [한국경제 - 인공지능 센서 장착하자 불량품이 사라졌다](https://n.news.naver.com/mnews/article/015/0004671424?sid=101)

> [ETRI - 자율주행이 불러올 새로운 이동수단의 가치](https://www.etri.re.kr/webzine/20190705/sub01.html)

DACON에 이와 관련된 대회가 있어 해당 대회에 참가하여 제공되는 데이터를 바탕으로 머신러닝기반의 성능예측 모델을 개발했습니다.
> [DACON - 자율주행 센서의 안테나 성능 예측 AI 경진대회](https://dacon.io/competitions/official/235927/overview/description)

## 데이터 분석
### X Features (56개)
![X Features](https://user-images.githubusercontent.com/100823210/188443938-b2b54706-6b31-4b32-a81c-d189415fae5d.png)

* 제조공정에서 수집된 데이터. (학습데이터 기준 39,607건)
* 검사통과여부 관련 값은 정수형(Int). 그 외 다른값은 실수형(Float)
* Null값은 존재하지 않음.
* 컬럼 카테고리
  * PCB 체결시 단계별 누름량
  * 방열재료 무게 및 면적
  * 검사 통과 여부
  * 커넥터 위치 기준 좌표
  * 각 안테나 패드 위치
  * 투입 전 대기 시간
  * 스크류 삽입 깊이 및 분당 회전수
  * 커넥터 핀 치수
  * 하우징 PCB 안착부 치수
  * 레이돔 치수
  * 안테나 부분 레이돔 기울기
  * RF 부분 SMT 납 량


### Y Targets (14개)
![Y Targets](https://user-images.githubusercontent.com/100823210/188444315-d80e3ea7-e5fe-40bc-906f-bf1f2351977d.png)
* 생산된 안테나의 성능을 나타내는 데이터 (14건)
* 모든값이 실수형(Float)
* Null값은 존재하지 않음.
* 컬럼 카테고리
  * 안테나 Gain 평균 및 편차
  * (평균) 신호대 잡음비

### 반복되는 패턴 발생
X축을 인덱스 기준으로 데이터를 나열하면, 아래와 같이 일정한 패턴을 보여주는 Feature들이 존재함. 즉, 조립과정에서 사용되는 기계의 동작이 일정한 Cycle을 가지고 있으며, 작은 Cycle이 6번 반복후에 좀 더 긴 Cycle이 6번 반복되는 모습을 보여주고 있음.
![X_03_방열재료1무게_인덱스기준](https://user-images.githubusercontent.com/100823210/188358475-609a879b-c73c-4b7a-afdd-5d71f45ccb97.png)

## 데이터 전처리
### Log Scaler
아래 방열재료의 면적(X_07 ~ X_09)과, 투입 전 대기시간(X_49)의 경우, 심하게 한쪽으로 치우친 분포를 보여 np.log1p를 적용.
![Log Scaler](https://user-images.githubusercontent.com/100823210/188358677-fc2553c4-e780-47b3-a762-28b3c221066f.png)

### 반복되는 패턴에 따른 Cycle의 고유값 부여
반복되는 패턴마다 푸리에 특징을 사용하여 간단한 요소로 분해하고 데이터가 연속성을 가질 수 있도록 표현하기 위하여 인덱스를 Cycle의 시작마다 0로 리셋한 값을 다시 np.sin, np.cos 함수를 적용하여 Cycle의 특징을 잘 나타내도록 변환한 후에 추가합니다.
![Cycle내 인덱스 고유값](https://user-images.githubusercontent.com/100823210/188359026-9f59a943-fee0-4267-8be8-28450755939d.png)

## 이상치(Outlier) 제거
앞서 데이터가 반복되는 특징을 가지고 있으며, Outlier도 이러한 Cycle에 영향을 받을 것으로 판단하여 지수평활법(ExponentialSmoothing)에서 Seasonality까지 고려하는 Holt-Winters 모델을 이용하여 Outlier 설정하고 제거해 보았습니다.
![Outlier 제거](https://user-images.githubusercontent.com/100823210/188359357-f766f6e9-dee4-4dd5-8724-8c80c543502e.png)

## 결론 및 한계점
### 결론
1. 다중출력(Multioutput) 문제를 푸는데 Estimator를 여러 개를 쓰는 것이 각각의 문제에 더 나은 결과를 가져올 수 있습니다. 하지만, 연산량이 많아지므로 테스트 1개 Cycle을 수행하는데 시간이 더 소모됩니다. 또한, 고려해야할 부분도 늘어나게 됩니다.
2. 문제를 복잡하게 바라보지 말고 최대한 단순하게 쪼개어서 봐야 한다는 점을 체감할 수 있었습니다. 전체 Target을 한꺼번에 처리하는 것보다 우선순위를 두어 하나의 Target의 문제만 처리하려고 했을 때, 좋은 결과를 가져올 수 있었습니다.
3. 데이터를 가능한 여러가지 방법으로 차트로 그려보는 것이 문제를 풀어가는데 도움이 되었습니다.
4. 기계가 발생시키는 센서 데이터의 경우, 반복되는 패턴이 발견될 수 있으며, 이를 이용하면 학습과 예측에 도움이 되는것을 확인 할 수 있었습니다.
5. 이상치(Outlier)를 과도하게 제거하면 Overfitting 문제가 발생될 수 있다는 점을 확인했습니다. 따라서 적당한 이상치 제거기준을 선택하기 위해서는 검증 데이터셋을 올바르게 설정하고 많은 테스트를 거쳐서 결정해야 합니다

### 한계점
1. 도메인 지식(라이다 센서, 제조공정 특성)에 대한 이해가 부족하면, 데이터를 해석하고 조합하는데 많은 어려움이 따른 다는것을 확인했습니다.
2. scikit-learn에는 다양한 함수들을 제공하고 있으며, 아는만큼 적용할 수 있다는 점에서 경험의 부족함을 느꼈습니다.
3. Feature에 대한 탐색을 제대로 하지 않으면 구현하는 과정에서 발생하는 선택을 자신이 신뢰하지 못하는 모습을 발견했습니다.
4. 대회라는 경쟁방식속에서 능동적으로 새로운 것을 찾고 적용해보려고 하는 과정은 좋았으나, Score에 집착하여 배운것을 정리하고 쌓아나가는 데는 실패했습니다.
5. 최초 프로젝트 방향은 머신러닝으로 Baseline을 설정하고, 딥러닝을 이용하여 문제를 해결하려고 하였으나, 모델 구성과 학습정도를 조절하는데 어려움을 느껴 포기하게 된 점이 아쉬웠습니다.


## 참조문헌
* [시계열 데이터 전처리(Encoding Time Step Features)](https://today-1.tistory.com/55)
* [마고커 - Exponential Smoothing을 활용한 이상탐지](https://magoker.tistory.com/120) 
* [제조 공정에서의 실시간 불량 탐지를 위한 딥러닝 모델 적용 연구](http://ktappi.kr/xml/30929/30929.pdf)
* [반도체 설비 센서 데이터를 활용한 딥러닝 기반의불량예측 모델에 관한 연구](https://www.koreascience.or.kr/article/CFKO202125036042269.pdf)
* [제조 공정에서 센서와 머신러닝을 활용한 불량예측 방안에 대한 연구](http://entrue.com/files/[4_2]%2089-98P_%EC%A0%9C%EC%A1%B0%20%EA%B3%B5%EC%A0%95_%ED%95%9C%EB%AC%B4%EB%AA%85%EC%B4%88.pdf)
* [기계학습 알고리즘을 활용한 베어링의 고장 예측 알고리즘 개발에 관한 연구](https://e-jamet.org/_common/do.php?a=full&b=52&bidx=1652&aidx=20609)

## 팀원 소개 
|팀원|연락|
|------|---|
|김송현|[G.mail](zpaladin1213@gmail.com) │ [Velog](https://velog.io/@zbooster)|
|김해솔|[G.mail](lunchtime99@gmail.com) │ [Velog](https://velog.io/@kim_haesol)|

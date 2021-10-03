# AI CCTV System Child Abuse Detection

2021 빅데이터 청년인재 고려대과정 10조 - 괴도키즈
---


- **프로젝트 개요** : CNN + BiLSTM을 통한 CCTV 실시간 아동학대 감지 시스템

- **프로젝트 작업 기간** : 2021.07.04 ~ 2021.08.28

- **팀원** : 김규민, 김규원, 김석기, 김호준, 이은서


![Logo](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/26e41d62-5d22-48e0-ad00-712422ba4884/%EA%B4%B4%EB%8F%84%ED%80%B4%EC%A6%88.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210828%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210828T004045Z&X-Amz-Expires=86400&X-Amz-Signature=ba5ca45a58050060fd58e6fc02b63d7a1eb68909bdb626ca2267295647bd1932&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22%25EA%25B4%25B4%25EB%258F%2584%25ED%2580%25B4%25EC%25A6%2588.png%22)

# Main Feature

1. 아동학대 지속을 방지하고, 즉시 아이를 보호할 수 있도록 **실시간 아동학대 감지 및 신고프로젝트**
2. 폭력이 일어나면 관할 경찰서에 신고하고, 학대 정황이 모호하다고 판단하면 그 영상을 **전문가에 의해 학대 정황이 확인**

# Service Flow

![service flow](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/da14f93b-2785-426a-a980-ed89dcf59966/%E3%84%B4%E3%84%B7%E3%84%B1.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210828%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210828T004037Z&X-Amz-Expires=86400&X-Amz-Signature=b374bc6ace620946b6d7bcecf5933e91d9ab3705643c13796f490c2195f181cb&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22%25E3%2584%25B4%25E3%2584%25B7%25E3%2584%25B1.png%22)


# System Architecture

![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/90ea0c7c-8e37-4940-8dc2-ce52bc4630f7/%EA%B7%B8%EB%A6%BC2.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210828%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210828T004104Z&X-Amz-Expires=86400&X-Amz-Signature=b9d047fa5a5a296cbdd4aeee4ac60be5797bec1e83399b6be2462ebcea6f2f79&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22%25EA%25B7%25B8%25EB%25A6%25BC2.png%22)

# Repository


```
└── biconvlstm_model
└── data
└── deep-learning-server
└── presentation
└── web
└── webcam live test
```

작성 : [김석기](https://github.com/ikgoes)

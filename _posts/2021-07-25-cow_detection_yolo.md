# YOLOv4를 이용한 이미지 데이터 증가 및 라벨링 전파

안녕하세요.

프로젝트 진행 중 기초 데이터가 부족해서 영상 데이터를 이용한 이미지 데이터 증가 방법에 대해 진행했었습니다.

특정 객체의 이미지 데이터만 추출 할 것이기 때문에 YOLOv4를 사용했습니다.

YOLO란 'You Only Look Once'의 약자로 실시간으로 객체를 탐지하는 기술입니다.

YOLO를 사용해 영상 데이터에서 실시간으로 탐지되는 객체를 자를 것입니다.

<그림 1>은 YOLO가 객체를 검출하는 과정을 나타낸 것입니다. YOLO의 큰 특징 중 하나는 이미지 전부로부터 특성을 뽑아서 각 bounding box를 예측하는 것입니다.

우선 입력 이미지 데이터를 SxS grid로 나누고 각각의 grid cell은 B개의 bounding box와 bounding box의 confidence score를 갖습니다. bounding box는 객체의 중심점, 높이와 넓이, 객체일 확률, box confidence score를 가지고 있습니다. confidence score는 $Pr(Object)*IOU$로 box가 특정 객체를 포함하고 있을 가능성을 계산해줍니다. 탐지된 객체가 특정 클래스에 포함될 확률은 $Pr(Class_{i}|Object)$로 계산한다. YOLO의 손실함수는 아래 식과 같다.

$\lambda_{coord}\displaystyle\sum_{i=0}^{S^2}\displaystyle\sum_{j=0}^{B}l_{ij}^{obj}[(x_i - \hat{x}_{i})^2 + (y_i - \hat{y}_{i})^2]$

$\quad +  \lambda_{coord}\displaystyle\sum_{i=0}^{S^2}\displaystyle\sum_{j=0}^{B}l_{ij}^{obj}[(\sqrt{w_i} - \sqrt{\hat{w}_{i}})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_{i}})^2]$

$\quad\quad\quad +\displaystyle\sum_{i=0}^{S^2}\displaystyle\sum_{j=0}^{B}l_{ij}^{obj}(C_i - \hat{C}_{i})^2$

$\quad\quad+\lambda_{noobj}\displaystyle\sum_{i=0}^{S^2}\displaystyle\sum_{j=0}^{B}l_{ij}^{noobj}(C_i - \hat{C}_{i})^2$

$\quad\quad\quad\quad+\displaystyle\sum_{i=0}^{S^2}l_{i}^{obj}\,\displaystyle\sum_{c∈classes}(p_i(c)-\hat{p}_{i}(c))^2$

Object가 존재하는 grid cell i의 predictor bounding box j에 대해, 중심점 x, y의 loss를 계산하고 Object가 존재하는 grid cell i의 predictor bounding box j에 대해, w(너비)와 h(높이)의 loss를 계산합니다. $C_i$는 confidenct socre의 loss 값을 계산합니다.

COCO dataset으로 사전 훈련된 YOLO를 이용해서 특정 객체를 인식하는 bounding box를 잘라서 특정 객체의 사진만 뽑아 냈습니다.

우선 영상을 input 데이터로 넣어 object detection 하는 함수는 'image_opencv.cpp' 파일에 있다.

'image_opencv.cpp' 파일의 'draw_detections_cv_v3' 함수를 이용해서 영상을 프레임별로 읽어오면 해당 프레임에서 object detection을 해주고 box confidence score를 표시해 줍니다.

이미지를 자르기 위해서는 아래에 나오는 코드들에 add code라고 주석 처리되어있는 코드를 해당 위치에 추가해주어야 한다.

추가된 코드에 대해 설명하면서 추가한 이유에 대해 설명하도록 하겠습니다.


```python
extern "C" void draw_detections_cv_v3(mat_cv* mat, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output)
{
    try {
        cv::Mat *show_img = (cv::Mat*)mat;
        int i, j;
    	int accur ; //add code(object abbuacy)
    	char img_name[100] ; //add code(file name)

        if (!show_img) return;
        static int frame_id = 0;
        frame_id++;

    	/*add code(cutting image) start*/
    	cv::Mat _image = *show_img ; //original frame image
    	cv::Mat subImage = _image.clone() ; //original frame image
    	/*add code end*/
```

* accure 변수 : box confidence score로 bounding box에 있는 객체의 정확도를 저장해 줍니다.
* img_name[100] : 저장할 이미지의 이름을 기록할 변수입니다. yolo는 c언어를 기본으로 짜여져 있기에 문자열의 길이를 지정해 줬다.

새로운 값을 저장할 변수를 지정하였습니다.

    /*add code(cutting image) start*/
    cv::Mat _image = *show_img ;
    cv::Mat subImage = _image.clone() ;

위 코드는 이미지를 자를 사진입니다. object detection을 하기 전에 이미지를 미리 복사를 하여 다른 객체의 bounding box의 선이 겹쳐서 잘리지 않게 미리 해당 프레임을 복사해 두었습니다.

해당 코드를 통해 subImage는 해당 프레임의 원본 이미지를 얻을 수 있습니다.


```python
for (j = 0; j < classes; ++j) {
    int show = strncmp(names[j], "dont_show", 9);
    if (dets[i].prob[j] > thresh && show) {
        if (class_id < 0) {
            strcat(labelstr, names[j]);
            class_id = j;
            char buff[20];
            if (dets[i].track_id) {
                sprintf(buff, " (id: %d)", dets[i].track_id);
                strcat(labelstr, buff);
            }
            sprintf(buff, " (%2.0f%%)", dets[i].prob[j] * 100);
            strcat(labelstr, buff);
            printf("%s: %.0f%% ", names[j], dets[i].prob[j] * 100);
            accur = dets[i].prob[j] * 100 ; //add code(object accuracy)
            sprintf(img_name, "%s", names[j]) ; //add code(object name)
            if (dets[i].track_id) printf("(track = %d, sim = %f) ", dets[i].track_id, dets[i].sim);
        }
        else {
            strcat(labelstr, ", ");
            strcat(labelstr, names[j]);
            printf(", %s: %.0f%% ", names[j], dets[i].prob[j] * 100);
            accur = dets[i].prob[j] * 100 ; //add code(objec accuracy)
            sprintf(img_name, "%s", names[j]) ; //add code(object name)
        }
    }
}
```

위 코드는 객체들을 인식하고 이름과 확률을 계산해 주는 과정을 담고 있습니다.

이 과정에서 객체의 이름(names[j])과 해당 객체의 확률(accur)을 저장 할 것입니다.


```python
if (class_id >= 0) {
    int width = std::max(1.0f, show_img->rows * .002f);

    //if(0){
    //width = pow(prob, 1./2.)*10+1;
    //alphabet = 0;
    //}

    //printf("%d %s: %.0f%%\n", i, names[class_id], prob*100);
    int offset = class_id * 123457 % classes;
    float red = get_color(2, offset, classes);
    float green = get_color(1, offset, classes);
    float blue = get_color(0, offset, classes);
    float rgb[3];

    //width = prob*20+2;

    rgb[0] = red;
    rgb[1] = green;
    rgb[2] = blue;
    box b = dets[i].bbox;
    if (std::isnan(b.w) || std::isinf(b.w)) b.w = 0.5;
    if (std::isnan(b.h) || std::isinf(b.h)) b.h = 0.5;
    if (std::isnan(b.x) || std::isinf(b.x)) b.x = 0.5;
    if (std::isnan(b.y) || std::isinf(b.y)) b.y = 0.5;
    b.w = (b.w < 1) ? b.w : 1;
    b.h = (b.h < 1) ? b.h : 1;
    b.x = (b.x < 1) ? b.x : 1;
    b.y = (b.y < 1) ? b.y : 1;
    //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

    int left = (b.x - b.w / 2.)*show_img->cols;
    int right = (b.x + b.w / 2.)*show_img->cols;
    int top = (b.y - b.h / 2.)*show_img->rows;
    int bot = (b.y + b.h / 2.)*show_img->rows;

    if (left < 0) left = 0;
    if (right > show_img->cols - 1) right = show_img->cols - 1;
    if (top < 0) top = 0;
    if (bot > show_img->rows - 1) bot = show_img->rows - 1;

    //int b_x_center = (left + right) / 2;
    //int b_y_center = (top + bot) / 2;
    //int b_width = right - left;
    //int b_height = bot - top;
    //sprintf(labelstr, "%d x %d - w: %d, h: %d", b_x_center, b_y_center, b_width, b_height);

    /*add code(cutting image) start*/
    if(accur >= 90 && frame_id % 125 == 0 && strcmp(img_name,"cow")==0) {
    //object accuracy is over 90% and object name is 'cow'

        cv::Mat cutImage ; // saving image
        cutImage = subImage(cv::Range(top, bot), cv::Range(left, right)) ; // image cut

        char filename[100] ;
        sprintf(filename, "/home/codingAlzi/cut_video/%d-%s-%d.jpg", frame_id, img_name, i) ;
        //file name and save point

        imwrite(filename, cutImage) ; //cutting image save the file
    }
    /*add code end*/
```

위 코드는 bounding box를 생성하는 코드입니다.

여기서 제가 추가한 코드는 'add code(cutting image) start'부터 'add code end'까지 입니다.

add code 위에서 객체 하나 당 bounding box를 하나 씩 생성해줍니다.

    if ( accur >= 90 && frame_id % 125 == 0 && strcmp(img_name,"cow")==0 )

이 코드를 통해 이 객체가 정확도가 90 이상이고 객체 이름이 'cow'인 객체인 것만 찾습니다.

그리고 frame_id % == 0인 이유는 매초 단위면 객체가 너무 많이 생성되어 메모리가 모자르기 때문에 5초마다 잘라주기 위해 했습니다.

(1초에 25프레임으로 계산했습니다.)

    cv::Mat cutImage ;
    cutImage = subImage(cv::Range(top, bot), cv::Range(left, right))

이 코드는 우선 자른 이미지를 저장할 변수 cutImage를 생성해줍니다.

cutImage에는 처음에 만든 원본 프레임 이미지 subImage에서 객체의 bounding box를 잘른 이미지를 저장해줍니다.

    char filename[100] ;
    sprintf(filename, "/home/codingAlzi/cut_video/%d=%s-%d.jpg", frame_id, img_name, i) ;
    
이 코드는 이미지를 저장할 위치와 이름을 지정하는 곳입니다.

sprintf()를 사용해서 filename이라는 변수에 'save/your/path/and/name'을 입력해주면 해당 위치에 저장됩니다.

저는 저장 파일명은 frame_id와 img_name과 for문의 i를 이용해서 이름이 겹치는 일이 없이 만들어 주었습니다.

    imsrite(filename, cutImage) ;
   
마지막 코드입니다. 이 코드는 자른 이미지가 저장된 cutImage를 filename의 위치에 저장하라는 명령어입니다.

오늘은 YOLO를 통해 영상에서 특정 객체를 bounding box를 기준으로 이미지를 잘라내는 방법에 대해 기술해 봤습니다.

우선 이 방법을 통해 특정 객체의 이미지를 자동으로 더 추출할 수 있을 것입니다.

다음엔 이미지 군집 및 라벨 전파 훈련에 대해 작성하겠습니다.

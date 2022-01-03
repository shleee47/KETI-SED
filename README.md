# Korea Electronics Technology Institute   
### Sound Event Dectection
   
   
## Github Installation   
### 1. Install github using the command below.
#### 아래 명령어를 통해서 깃허브를 설치해주세요.
```
$ sudo apt install git
$ git config --global user.name "USER NAME"
$ git config --global user.email "EMAIL@gmail.com"   
```      

   
## Preparation    
### 1. Get the code on the server.
#### 서버에 코드를 받으세요.  
```
$ git clone https://github.com/shleee47/RaspberryPi-boar-detection.git
```     
### 2. Create the environment.   
#### 환경을 만들어주세요.   
```
KETI-SED/  
$ sh environment.sh
```      

### 3. Download the model weights from the drive and place them in the path below.
#### 구글 드라이브에서 모델을 다운로드 하고 아래 경로에 위치시키세요.
https://drive.google.com/file/d/1zaOprKw6-s4CnvLG1USxBrDlOTdw2gMW/view?usp=sharing
```
KETI-SED/  
  └── dataset/
    └── wav
        └── Explosion
        └── Speech
            .....
            .....
            .....
        └── Pig

```            

### 4. Choose the event class you want to train from the path below.
#### 아래 경로에서 학습할 이벤트 클래스를 정하세요.
```
KETI-SED/
$ dataloader.py
  └── line 80 ~ 87
  └── line 153 ~ 157
```          

### 5. Place the csv file of the event class you want to train in the path below.
```
KETI-SED/  
  └── train/
    └── yes.csv
    └── no.csv
```     

### 6. Run main.sh for train
```
KETI-SED/  
$ sh main.sh
```            

   

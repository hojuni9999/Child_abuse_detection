import os
import time
import shutil
import json
import hashlib
import datetime
import numpy as np

import torch
import torch.optim
import torch.utils.data

from model import add_video_db, add_report_db, Video, ReportList, DaycareCenter, Location, User
from flask_sqlalchemy import SQLAlchemy
from flask_ngrok import run_with_ngrok
from flask import Flask, url_for, redirect, render_template, request, flash
from dl_model import load_model, data_loader
from data.data_reader import read_video

video_status = {'need_check':'0', 'reported':'1', 'safe':'2'}

model_path = '/content/drive/Shareddrives/2021청년인재_고려대과정_10조/Test Data/호준/호준model_best.E_bi_max_pool_ALL_fold_0_t2021-08-27 01:04:46.tar'

args = {
    'arch': 'E_bi_max_pool',
    'workers':1,
    'batch-size':1,
    'evalmodel':model_path,
    'kfold':0,
    'split':5,
    'frames':5,
    'lr':1e-6,
    'weight_decay':1e-1
}

model = load_model(args,model_path)        
daycare_center_name = '예은어린이집'
violence_threshold = 90
uncertain_threshold = 80

# 영상 저장을 위한 변수
save_video_path = '/content/drive/Shareddrives/2021청년인재_고려대과정_10조/Web/static/saved video/'
save_violence_path = '/content/drive/Shareddrives/2021청년인재_고려대과정_10조/Web/static/violence/'
save_uncertain_path = '/content/drive/Shareddrives/2021청년인재_고려대과정_10조/Web/static/uncertain/'

app = Flask(__name__,static_folder='static',template_folder='templates')
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///child_abuse_detection_database.db'
db = SQLAlchemy(app)
run_with_ngrok(app)   #starts ngrok when the app is run

def get_hashed_password(password):
    h = hashlib.sha256()
    h.update(password.encode('ascii'))
    return h.hexdigest()

def check_email(email):
    result = User.query.filter_by(email=email).all()
    return True if result else False

def convert_datetime(unixtime):
    """Convert unixtime to datetime"""
    date = datetime.datetime.fromtimestamp(unixtime).strftime('%Y-%m-%d %H_%M_%S')
    return date
    
def convert_unixtime(date_time):
    """Convert datetime to unixtime"""
    unixtime = datetime.datetime.strptime(date_time,'%Y-%m-%d %H_%M_%S').timestamp()
    return unixtime

def check_login(email, pw):
    result = User.query.filter_by(email=email, pw=pw).all()
    return (False, None, None) if not result else (True, result[0].id ,result[0].loc_id)

@app.route('/index')
@app.route('/')
def home():
    return render_template('index.html')

current_user = None
current_location = None
@app.route('/main',methods=['GET','POST'])
def maain():
    global current_user, current_location
    if request.method == 'GET':
        return render_template('main.html')
    else:
        email = request.form['email'] 
        password = get_hashed_password(request.form['password'])
        status, current_user, current_location = check_login(email,password)
        if status:
            return render_template('main.html')
        else:
            flash("please put correct email or password")
            return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        name = request.form.get('name')
        password = get_hashed_password(request.form.get('password'))
        email = request.form.get('email')
        officenum = request.form.get('officenum')
        location1 = request.form.get('locate1')
        location2 = request.form.get('locate2')
        department = request.form.get('dept')
        ph1 = request.form.get('ph_num1')
        ph2 = request.form.get('ph_num2')
        ph3 = request.form.get('ph_num3')
        if check_email(email):
            flash('email already exists')
            return render_template('register.html')
        elif not (name and password and email and officenum and location1 and location2 and department and ph1 and ph2 and ph3):
            flash('fill all the area')
            return render_template('register.html')
        else:
            location_id = Location.query.filter(Location.name==location1+' '+location2).one().id
            user = User(
                email=email,
                pw=password,
                office_num=officenum,
                department=department,
                name=name,
                ph_num1=ph1,
                ph_num2=ph2,
                ph_num3=ph3,
                loc_id = location_id)
            db.session.add(user)
            db.session.commit()
            flash('register finished')
            return render_template('index.html')

@app.route('/video')
def video():
    # get data by current_location
    video_data = Video.query.filter_by(loc_id=current_location, status=video_status['need_check']).all()

    data = list()
    for item in video_data:
        daycare_info = DaycareCenter.query.filter_by(id=item.dc_id).one() 
        daycare_description = "어린이집명 : {}<br>원장 : {}<br>주소 : {}<br>전화번호 : {}-{}-{}".format(daycare_info.name,daycare_info.chief_staff_name,daycare_info.address, daycare_info.ph_num1,daycare_info.ph_num2,daycare_info.ph_num3)
        data.append({
            "index":item.id, 
            "place":daycare_info.name,
            "time":convert_datetime(item.detection_time),
            "time_unix":item.detection_time,
            "accuracy":str(item.accuracy) + ' %',
            "accuracy_":item.accuracy,
            "video_path":item.name,
            "video_info":daycare_description          
            })

    acc_sorted_data = sorted(data, key=lambda x: int(x['accuracy_']), reverse=True)
    time_sorted_data = sorted(data, key=lambda x:int(x['time_unix']), reverse=True)
    return render_template('video.html', 
                            acc_sorted_data=acc_sorted_data,
                            time_sorted_data=time_sorted_data,
                            data_length=len(video_data), 
                            video_info="Video Description") 

@app.route('/list')
def listing():
    report_data = ReportList.query.filter_by(loc_id=current_location).all()
    data = list()
    
    for item in report_data:
        daycare_info = DaycareCenter.query.filter_by(id=item.dc_id).one()
        video_info = Video.query.filter_by(id=item.vid_id).one()
        daycare_description = "영상 정확도 : {}<br>어린이집명 : {}<br>원장 : {}<br>주소 : {}<br>전화번호 : {}-{}-{}".format(str(video_info.accuracy) + '%', daycare_info.name,daycare_info.chief_staff_name,daycare_info.address, daycare_info.ph_num1,daycare_info.ph_num2,daycare_info.ph_num3)
        data.append({
            "index":item.id, 
            "daycare":daycare_info.name,
            "report_time":convert_datetime(item.time),
            "time_unix":item.time,
            "action_time": convert_datetime(video_info.detection_time),
            "video_path": video_info.name,
            "video_info":daycare_description,
            "police_station":item.police_name,
            "police_status":item.status      
            })

    time_sorted_data = sorted(data, key=lambda x:int(x['time_unix']), reverse=True)
    return render_template('list.html', 
                            time_sorted_data=time_sorted_data,
                            data_length=len(time_sorted_data), 
                            video_info="Video Description"
                            ) 

@app.route('/report/<video_id>')
def report_police(video_id):
    police_station = ['답십리지구대', '용신지구대', '청량리파출소', '제기파출소', '전농1파출소', '전농2파출소','장안1파출소', '장안2파출소', '이문지구대', '휘경파출소', '회기파출소']
    video = db.session.query(Video).get(int(video_id))
    video.status = video_status['reported']

    report_data = ReportList(
        time = time.time() + 9 * 3600, 
        police_name = np.random.choice(police_station, 1)[0],
        status = '출동 전',
        loc_id = video.loc_id,
        dc_id = video.dc_id,
        vid_id = video.id
    )
    db.session.add(report_data)
    db.session.commit()
    return redirect(url_for('video'))

@app.route('/safe/<video_id>')
def safe_video(video_id):
    video = db.session.query(Video).get(int(video_id))
    video.status = video_status['safe']
    db.session.commit()
    return redirect(url_for('video'))

@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        current_time = time.time()+9*60*60
        end = time.time()

        file = request.files['file']
        frames = file.read()
        
        FILE_OUTPUT = daycare_center_name + '_' + str(time.strftime('%Y-%m-%d_%H_%M_%S %p', time.gmtime(current_time))) + '.mp4'

        out_file = open(save_video_path + FILE_OUTPUT, "wb") 
        out_file.write(frames)
        out_file.close()

        video_path = save_video_path + FILE_OUTPUT
        
        video = read_video(video_path)
        val_loader = data_loader(args, video)

        model.eval()

        for i, (input) in enumerate(val_loader):
            input_var = torch.autograd.Variable(input, volatile=True)
            input_var = input_var.cuda()

            # compute output
            output_dict = model(input_var)

            model_ret = output_dict['classification'].cpu().detach().numpy()[0][1] * 100
            
            # Violence
            if model_ret > violence_threshold:
                print('-----------------violence detected!!!!!!----------------')
                print(model_ret, '% violence detected')

                shutil.copy(save_video_path+FILE_OUTPUT, save_violence_path+FILE_OUTPUT)
                save_name = save_violence_path+FILE_OUTPUT.split('.')[0] + '_' + str(round(model_ret,2)) +'.mp4'
                os.rename( save_violence_path+FILE_OUTPUT, save_name)

                video_info = add_video_db(db, save_name, daycare_center_name, model_ret, status=1)
                report_info = add_report_db(db,video_info)

            # Uncertain
            elif model_ret > uncertain_threshold:
                print('****** uncertainty detected ******')
                print(model_ret, '% violence detected')

                shutil.copy(save_video_path+FILE_OUTPUT, save_uncertain_path+FILE_OUTPUT)
                save_name = save_uncertain_path+FILE_OUTPUT.split('.')[0] + '_' + str(round(model_ret,2)) +'.mp4'
                os.rename( save_uncertain_path+FILE_OUTPUT,  save_name)

                video_info = add_video_db(db, save_name,daycare_center_name, model_ret)
            
            else:
                print('violence : ', round(model_ret,2),' %, - ',FILE_OUTPUT)
                os.rename(save_video_path+FILE_OUTPUT, save_video_path+FILE_OUTPUT.split('.')[0] + '_' + str(round(model_ret,2)) +'.mp4')

        print('model calulation time : ',round(time.time() - end,2), ' sec')

        if model_ret > violence_threshold:  
            return json.dumps(str(1))
        else:
            return json.dumps(str(0))

if __name__ == "__main__":
    app.run()


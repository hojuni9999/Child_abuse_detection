import re
import os
import time
import shutil
import hashlib
import datetime
import numpy as np
from flask import Flask, url_for, redirect, render_template, request
from flask_login import current_user
from flask_sqlalchemy import SQLAlchemy
from flask import flash
from model import *

app = Flask(__name__,static_folder='static',template_folder='templates')
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///child_abuse_detection_database.db'
db = SQLAlchemy(app)
#run_with_ngrok(app)

def get_hashed_password(password):
    h = hashlib.sha256()
    h.update(password.encode('ascii'))
    return h.hexdigest()

def check_email(email):
    result = User.query.filter_by(email=email).all()
    print(result)
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
    print(email,pw)
    result = User.query.filter_by(email=email, pw=pw).all()
    print(result)
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
        print(password)
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

@app.route('/main')
def mainpage():
    return render_template('main.html')

@app.route('/video')
def video():
    # get data by current_location
    video_data = Video.query.filter_by(loc_id=current_location, status='0').all()
    data = list()
    for item in video_data:
        daycare_info = DaycareCenter.query.filter_by(id=item.dc_id).one() # 0: 확인해야함 / 1: 신고됨 / 2: 폭력없음
        daycare_description = "어린이집명 : {}<br>원장 : {}<br>주소 : {}<br>전화번호 : {}-{}-{}".format(daycare_info.name,daycare_info.chief_staff_name,daycare_info.address, daycare_info.ph_num1,daycare_info.ph_num2,daycare_info.ph_num3)
        data.append({
            "index":item.id, 
            "place":daycare_info.name,
            "time":convert_datetime(item.detection_time),
            "time_unix":item.detection_time,
            "accuracy":str(item.accuracy) + ' %',
            "accuracy_":item.accuracy,
            "video_path":"/static/uncertain/" + item.name,
            "video_info":daycare_description          
            })

    acc_sorted_data = sorted(data, key=lambda x: int(x['accuracy_']), reverse=True)
    time_sorted_data = sorted(data, key=lambda x:int(x['time_unix']), reverse=True)
    return render_template('video.html', 
                            acc_sorted_data=acc_sorted_data,
                            time_sorted_data=time_sorted_data,
                            data_length=len(video_data), 
                            video_info="Video Description"
                            ) 

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
            "video_path":"/static/violence/" + video_info.name,
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
    # report 처리
    police_station = ['답십리지구대', '용신지구대', '청량리파출소', '제기파출소', '전농1파출소', '전농2파출소','장안1파출소', '장안2파출소', '이문지구대', '휘경파출소', '회기파출소']
    video = db.session.query(Video).get(int(video_id))
    video.status = '1'

    if os.path.isfile("/static/uncertain/" + video.name):
        print('move ', video.name)
        shutil.copy("/static/uncertain/" + video.name, "/static/violence/" + video.name)

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
    # safe 처리
    video = db.session.query(Video).get(int(video_id))
    video.status = '2'
    db.session.commit()
    return redirect(url_for('video'))
    
if __name__ == '__main__':
      app.run()


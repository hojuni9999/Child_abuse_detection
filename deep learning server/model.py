# coding: utf-8
import os
import time
import numpy as np
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()



class DaycareCenter(db.Model):
    __tablename__ = 'daycare_center'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Integer, nullable=False)
    chief_staff_name = db.Column(db.String, nullable=False)
    address = db.Column(db.String, nullable=False)
    ph_num1 = db.Column(db.String, nullable=False)
    ph_num2 = db.Column(db.String, nullable=False)
    ph_num3 = db.Column(db.String, nullable=False)
    loc_id = db.Column(db.ForeignKey('location.id'))

    loc = db.relationship('Location', primaryjoin='DaycareCenter.loc_id == Location.id', backref='daycare_centers')



class Location(db.Model):
    __tablename__ = 'location'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Integer, nullable=False)



class ReportList(db.Model):
    __tablename__ = 'report_list'

    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.Integer, nullable=False)
    police_name = db.Column(db.String, nullable=False)
    status = db.Column(db.String, nullable=False)
    loc_id = db.Column(db.ForeignKey('location.id'))
    dc_id = db.Column(db.ForeignKey('daycare_center.id'))
    vid_id = db.Column(db.ForeignKey('video.id'))

    dc = db.relationship('DaycareCenter', primaryjoin='ReportList.dc_id == DaycareCenter.id', backref='report_lists')
    loc = db.relationship('Location', primaryjoin='ReportList.loc_id == Location.id', backref='report_lists')
    vid = db.relationship('Video', primaryjoin='ReportList.vid_id == Video.id', backref='report_lists')



class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String, nullable=False)
    pw = db.Column(db.String, nullable=False)
    office_num = db.Column(db.Integer, nullable=False)
    department = db.Column(db.String, nullable=False)
    name = db.Column(db.String, nullable=False)
    ph_num1 = db.Column(db.String, nullable=False)
    ph_num2 = db.Column(db.String, nullable=False)
    ph_num3 = db.Column(db.String, nullable=False)
    loc_id = db.Column(db.ForeignKey('location.id'))

    loc = db.relationship('Location', primaryjoin='User.loc_id == Location.id', backref='users')



class Video(db.Model):
    __tablename__ = 'video'

    id = db.Column(db.Integer, primary_key=True)
    detection_time = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    status = db.Column(db.String, nullable=False)
    loc_id = db.Column(db.ForeignKey('location.id'))
    dc_id = db.Column(db.ForeignKey('daycare_center.id'))

    dc = db.relationship('DaycareCenter', primaryjoin='Video.dc_id == DaycareCenter.id', backref='videos')
    loc = db.relationship('Location', primaryjoin='Video.loc_id == Location.id', backref='videos')

def add_video_db(db, video_path, daycare_name, accuracy, status=0):
    daycare_center = DaycareCenter.query.filter_by(name=daycare_name).one()

    video = Video(
        detection_time = time.time() + 9 * 3600,
        name = os.path.basename(video_path),
        accuracy = round(accuracy,2),
        status = status,
        loc_id = daycare_center.loc_id,
        dc_id = daycare_center.id
        )

    db.session.add(video)
    db.session.commit()
    return video

def add_report_db(db, video_info):
    police_station = ['답십리지구대', '용신지구대', '청량리파출소', '제기파출소', '전농1파출소', '전농2파출소','장안1파출소', '장안2파출소', '이문지구대', '휘경파출소', '회기파출소']
    
    video = Video.query.filter_by(name=video_info.name).one()
    video.status = 1
    
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

    return report_data

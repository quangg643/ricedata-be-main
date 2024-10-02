from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Files(db.Model):
    __tablename__ = "files"
    id = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(200), nullable=False) 
    extension = db.Column(db.String(200), nullable=False)
    
    visualized_images = db.relationship('VisualizedImages', backref='files', lazy=True)
    recommend_channel = db.relationship('RecommendChannels', backref='files', lazy=True)
    


class VisualizedImages(db.Model):
    __tablename__ = "visualized_images"
    id = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    file_id = db.Column(db.Integer, db.ForeignKey('files.id'), nullable=False)
    visualized_filename = db.Column(db.String(255), nullable=False)
    visualized_filepath = db.Column(db.String(200), nullable=False)
    width = db.Column(db.Integer)  
    height = db.Column(db.Integer)  

    statistical_data = db.relationship('StatisticalData', backref='visualized_images', lazy=True)
    points = db.relationship('Points', backref='visualized_images', lazy=True)


class RecommendChannels(db.Model):
    __tablename__ = "recommend_channels"
    id = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    file_id = db.Column(db.Integer, db.ForeignKey('files.id'), nullable=False)
    R = db.Column(db.Integer, nullable=False)
    G = db.Column(db.Integer, nullable=False)
    B = db.Column(db.Integer, nullable=False)


class Points(db.Model):
    _tablename_ = "points"
    id = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    image_id = db.Column(db.Integer, db.ForeignKey('visualized_images.id'), nullable=False)
    point_id = db.Column(db.String(255), db.ForeignKey('statistical_data.point_id'), nullable=False)
    x = db.Column(db.Integer, nullable=False)
    y = db.Column(db.Integer, nullable=False)


class StatisticalData(db.Model):
    __tablename__ = "statistical_data"
    id = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    image_id = db.Column(db.Integer, db.ForeignKey('visualized_images.id'), nullable=False) 
    point_id = db.Column(db.String(255), nullable=False)
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    h = db.Column(db.Float, nullable=True)
    replicate = db.Column(db.String(255), nullable=True)
    sub_replicate = db.Column(db.Integer, nullable=True)
    chlorophyll = db.Column(db.Float, nullable=True)
    rice_height = db.Column(db.Float, nullable=True)   
    spectral_num = db.Column(db.Integer, nullable=True)
    digesion = db.Column(db.Float, nullable=True)
    p_conc = db.Column(db.Float, nullable=True)
    k_conc = db.Column(db.Float, nullable=True)
    n_conc = db.Column(db.Float, nullable=True)
    chlorophyll_a = db.Column(db.Float, nullable=True)
    date  = db.Column(db.Date, nullable=False)

    points = db.relationship('Points', backref='statistical_data', lazy=True)

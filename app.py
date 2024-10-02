#             img_path = hsi_to_rgb(img_name, 55, 28, 12)
import logging
import traceback
import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from models import db, Files, Points, VisualizedImages, RecommendChannels, StatisticalData
import pandas as pd
import uuid
from PIL import Image
import spectral as sp

from npy_append_array import NpyAppendArray
import numpy as np

from pyproj import Proj, transform
import math

import matplotlib
import matplotlib.pyplot as plt
import specdal
from io import BytesIO


app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = 'nhatanhng'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ricedata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['ALLOWED_EXTENSIONS'] = {'csv'}


db.init_app(app)

logging.basicConfig(level=logging.DEBUG)

with app.app_context():
    db.create_all()

UPLOAD_FOLDER = 'uploads'
VISUALIZED_FOLDER = 'visualized'
UPLOAD_FOLDER_NPY = 'uploads/npy'
UPLOAD_CSV_FOLDER = 'uploads/csv_mapping_points'
UPLOAD_REFLECTANCE_DATA = 'uploads/reflectance_data'


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(VISUALIZED_FOLDER):
    os.makedirs(VISUALIZED_FOLDER)
if not os.path.exists(UPLOAD_FOLDER_NPY):
    os.makedirs(UPLOAD_FOLDER_NPY)
if not os.path.exists(UPLOAD_REFLECTANCE_DATA):
    os.makedirs(UPLOAD_REFLECTANCE_DATA)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VISUALIZED_FOLDER'] = VISUALIZED_FOLDER 
app.config['UPLOAD_FOLDER_NPY'] = UPLOAD_FOLDER_NPY
app.config['UPLOAD_CSV_FOLDER'] = UPLOAD_CSV_FOLDER
app.config['UPLOAD_REFLECTANCE_DATA'] = UPLOAD_REFLECTANCE_DATA


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def npy_converter(img):
    # store the image with its name and npy extension and save it in ./uploads/npy
    npy_filename = "./uploads/npy/" + img.filename.split('.')[0] + '.npy'
    hdr_name = "./uploads/" + img.filename.split('.')[0] + '.hdr'
    hdr_img = sp.envi.open(hdr_name)
    average = []
    try:
        with NpyAppendArray(npy_filename) as npy:
            for i in range(122):
                channel = np.expand_dims(hdr_img.read_band(i), 0)
                average.append(np.average(channel))
                npy.append(channel)

        blue = round(np.max(average[0:15]))
        green = round(np.max(average[16:40]))
        red = round(np.max(average[41:85]))
        # nf = np.max(average[86:121])

        file_record = Files.query.filter_by(filename=img.filename).first()
        if not file_record:
            raise ValueError("File not found in the database.")

        recommend_channel = RecommendChannels.query.filter_by(file_id=file_record.id).first()

        if recommend_channel:
            recommend_channel.R = red
            recommend_channel.G = green
            recommend_channel.B = blue
            db.session.commit()
        else:
            recommend_channel = RecommendChannels(
                file_id=file_record.id,
                R=red,
                G=green,
                B=blue,
            )
            db.session.add(recommend_channel)
            db.session.commit()
        
        db.session.add(recommend_channel)
        db.session.commit()

        return jsonify({"message": "File processed and data saved successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def hsi_to_rgb(hsi_img_name, red, green, blue):
    hsi_img = np.load(os.path.join(UPLOAD_FOLDER_NPY, hsi_img_name + '.npy'))

    red_band = hsi_img[red].astype(np.uint8)
    green_band = hsi_img[green].astype(np.uint8)
    blue_band = hsi_img[blue].astype(np.uint8)

    red_normalized = np.where(red_band > 50, 50, red_band)
    green_normalized = np.where(green_band > 50, 50, green_band)
    blue_normalized = np.where(blue_band > 50, 50, blue_band)

    dr_main_image = np.zeros((red_normalized.shape[0], red_normalized.shape[1], 3), dtype=np.uint8)
    dr_main_image[:, :, 0] = red_normalized
    dr_main_image[:, :, 1] = green_normalized
    dr_main_image[:, :, 2] = blue_normalized

    dr_main_image = (255 * (1.0 / dr_main_image.max() * (dr_main_image - dr_main_image.min()))).astype(np.uint8)

    main_image = Image.fromarray(dr_main_image)
    output_path = os.path.join(VISUALIZED_FOLDER, hsi_img_name + ".png")
    main_image = main_image.save(output_path)

    return output_path


def convert_to_pixels(northing, easting, base_northing, base_easting, original_width, original_height, display_width, display_height):
    scale_coordinate =  0.035

    # Convert to pixel coordinates based on the original image size
    x_pixel = abs(easting - base_easting) / scale_coordinate
    y_pixel = abs(base_northing - northing) / scale_coordinate

    # Rotation matrix for 45 degrees
    cos_45 = math.cos(math.radians(45))
    sin_45 = math.sin(math.radians(45))

    x_pixel_rotated = x_pixel * cos_45 - y_pixel * sin_45
    y_pixel_rotated = x_pixel * sin_45 + y_pixel * cos_45

    # Translate to ensure coordinates are positive
    x_pixel_rotated += original_width / 2
    y_pixel_rotated += original_height / 2

    # Scale to display size
    x_pixel_scaled = x_pixel_rotated * (display_width / original_width)
    y_pixel_scaled = y_pixel_rotated * (display_height / original_height)

    # Clip values to ensure they stay within display image dimensions
    x_pixel_clipped = max(0, min(int(x_pixel_scaled), display_width - 1)) + 240
    y_pixel_clipped = max(0, min(int(y_pixel_scaled), display_height - 1)) - 575


    return x_pixel_clipped, y_pixel_clipped

@app.route('/files', methods=['GET'])
def get_files():
    if request.method == 'GET':
        files = Files.query.all()
        return jsonify([{'id': file.id, 'filename': file.filename} for file in files]), 200

@app.route('/uploads/files', methods=['POST'])
def upload():
    try:
        if request.method == 'POST':
            file = request.files['file']
            if file:                
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_extension = filename.split('.')[-1]
                file.save(filepath)
                upload = Files(filename=filename, filepath=filepath, extension=file_extension)
                db.session.add(upload)
                db.session.commit()
                
                file_record = Files.query.filter_by(filename=filename).first()
                if file_record.extension == 'hdr':
                    npy_converter(file) 
                
                return f'Uploaded: {filename}'       
            else:
                return 'No file uploaded', 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/check-file', methods=['POST'])
def check_file():
    filenames = request.json.get('filenames', [])
    existing_files = Files.query.filter(Files.filename.in_(filenames)).all()
    if existing_files:
        existing_filenames = [file.filename for file in existing_files]
        return jsonify({"exists": True, "existsFiles": existing_filenames}), 200
    return jsonify({"exists": False}), 200

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    try:
        filename = secure_filename(filename)

        file = Files.query.filter_by(filename=filename).first()
        if not file:
            return jsonify({"error": "File not found"}), 404
        return send_file(file.filepath, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        file = Files.query.filter_by(filename=filename).first()
        if not file:
            return jsonify({"error": "File not found"}), 404
        

        visualized_images = VisualizedImages.query.filter_by(file_id=file.id).all()
        for visualized_image in visualized_images:
            points = Points.query.filter_by(image_id=visualized_image.id).all()
            for point in points:
                db.session.delete(point)
            
            statistical_data = StatisticalData.query.filter_by(image_id=visualized_image.id).all()
            for data in statistical_data:
                db.session.delete(data)
            
            db.session.delete(visualized_image)
        
        recommend_channels = RecommendChannels.query.filter_by(file_id=file.id).all()
        for recommend_channel in recommend_channels:
            db.session.delete(recommend_channel)
        
        db.session.delete(file)
        db.session.commit()

        if os.path.exists(file.filepath):
            os.remove(file.filepath)
        else:
            return jsonify({"message": f"File {filename} deleted from database, but file was not found on disk"}), 200
        
        return jsonify({"message": f"File {filename} deleted successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error deleting file {filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500        


@app.route('/rename/<filename>', methods=['PUT'])
def rename_file(filename):
    try:
        new_filename = request.json.get('newFilename')
        if not new_filename:
            return jsonify({"error": "New filename not provided"}), 400
        
        # Retrieve the file from the database
        file = Files.query.filter_by(filename=filename).first()
        if not file:
            return jsonify({"error": "File not found"}), 404

        # Generate new file paths
        new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(new_filename))

        # Rename the file on the filesystem
        os.rename(file.filepath, new_filepath)

        # Rename the associated .npy file (if it exists)
        old_npy_filename = filename.rsplit('.', 1)[0] + ".npy"
        old_npy_filepath = os.path.join(app.config['UPLOAD_FOLDER_NPY'], old_npy_filename)
        if os.path.exists(old_npy_filepath):
            new_npy_filename = new_filename.rsplit('.', 1)[0] + ".npy"
            new_npy_filepath = os.path.join(app.config['UPLOAD_FOLDER_NPY'], new_npy_filename)
            os.rename(old_npy_filepath, new_npy_filepath)

        # Rename the associated visualized image (if it exists)
        visualized_image = VisualizedImages.query.filter_by(file_id=file.id).first()
        if visualized_image:
            old_visualized_filepath = visualized_image.visualized_filepath
            if old_visualized_filepath:
                new_image_filename = new_filename.rsplit('.', 1)[0] + ".png"
                new_image_filepath = os.path.join(app.config['VISUALIZED_FOLDER'], new_image_filename)
                os.rename(old_visualized_filepath, new_image_filepath)

                # Update visualized image record in the database
                visualized_image.visualized_filename = new_image_filename
                visualized_image.visualized_filepath = new_image_filepath

        # Update the file record in the database
        file.filename = new_filename
        file.filepath = new_filepath

        # Commit the changes to the database
        db.session.commit()
        
        return jsonify({"message": f"File {filename} renamed to {new_filename} and related image renamed"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_image_from_hdr/<string:filename>', methods=['GET'])
def get_image_from_hdr(filename):
    # Retrieve the HDR file from the database using the filename
    hdr_file = Files.query.filter_by(filename=filename, extension='hdr').first()

    if not hdr_file:
        return jsonify({"error": "HDR file not found."}), 404

    # Find the corresponding visualized image using the file_id
    visualized_image = VisualizedImages.query.filter_by(file_id=hdr_file.id).first()

    if not visualized_image:
        return jsonify({"error": "No image has been created from this HDR file."}), 404

    # Check if the visualized file exists in the file system
    image_path = visualized_image.visualized_filepath
    if not os.path.exists(image_path):
        return jsonify({"error": "Visualized image file does not exist on the server."}), 404

    # Return the image file to the user
    return send_file(image_path, mimetype='image/png')


    
@app.route('/hyperspectral', methods=['POST'])
def visualize_HSI():
    try:
        data = request.json
        filename = data['filename']
        r = data['R']
        g = data['G']
        b = data['B'] 
        
        img_name = filename.split('.')[0]
        img_path = os.path.join(VISUALIZED_FOLDER, img_name + '.png')
        
        img_path = hsi_to_rgb(img_name, r, g, b)
        logging.info(f"Image {img_name}.png created and saved with R={r}, G={g}, B={b}.")

        # Open the image to get its dimensions
        with Image.open(img_path) as img:
            width, height = img.size

        
        file_record = Files.query.filter_by(filename=filename).first()
        if file_record:
            visualized_image = VisualizedImages.query.filter_by(file_id=file_record.id).first()
            if visualized_image:
                visualized_image.visualized_filepath = img_path
                visualized_image.width = width  
                visualized_image.height = height  
            else:
                visualized_image = VisualizedImages(
                    file_id=file_record.id,
                    visualized_filename=img_name + '.png',
                    visualized_filepath=img_path,
                    width=width,  
                    height=height  
                )
                db.session.add(visualized_image)
            db.session.commit()

        return send_file(img_path, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error visualizing hyperspectral image: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/recommend_channel/<filename>', methods=['GET'])
def get_recommend_channel(filename):
    print(f"Received request for filename: {filename}")

    file_record = Files.query.filter_by(filename=filename).first()
    if not file_record:
        print(f"File record not found for filename: {filename}")
        return jsonify({"error": "File not found"}), 404

    recommend_channel = RecommendChannels.query.filter_by(file_id=file_record.id).first()
    if not recommend_channel:
        print(f"Recommendation channel not found for filename: {file_record.filename}")
        return jsonify({"error": "Recommendation channel not found"}), 404

    print(f"Recommendation channel found: R={recommend_channel.R}, G={recommend_channel.G}, B={recommend_channel.B}")

    return jsonify({
        "R": recommend_channel.R,
        "G": recommend_channel.G,
        "B": recommend_channel.B
    })  


@app.route('/visualized_files', methods=['GET'])
def get_visualized_files():
    try:
        visualized_images = VisualizedImages.query.all()
        visualized_filenames = [img.visualized_filename for img in visualized_images]
        return jsonify(visualized_filenames), 200
    except Exception as e:
        logging.error(f"Error fetching visualized files: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/visualized/<filename>', methods=['GET'])
def get_visualized_file(filename):
    try:
        file_path = os.path.join(app.config['VISUALIZED_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        return send_file(file_path, mimetype='image/png')
    except Exception as e:
        logging.error(f"Error serving visualized file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/save_points/<filename>', methods=['POST'])
def save_points(filename):
    try:
        if not filename.endswith('.png'):
            filename = filename.rsplit('.', 1)[0] + '.png'

        print(filename)
        visualized_image = VisualizedImages.query.filter_by(visualized_filename=filename).first()
        if not visualized_image:
            return jsonify({"error": "File record not found"}), 404

        points = request.json.get('points', [])
        
        Points.query.filter_by(image_id=visualized_image.id).delete()

        for point in points:
            new_point = Points(
                image_id=visualized_image.id,
                x=point['x'],
                y=point['y']
            )
            db.session.add(new_point)

        db.session.commit()
        return jsonify({"message": "Points saved successfully"}), 200

    except Exception as e:
        logging.error(f"Error saving points: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/delete_point/<int:point_id>', methods=['DELETE'])
def delete_point(point_id):
    try:
        point = Points.query.get(point_id)
        if not point:
            return jsonify({"error": "Point not found"}), 404

        db.session.delete(point)
        db.session.commit()
        return jsonify({"message": "Point deleted successfully"}), 200
    except Exception as e:
        logging.error(f"Error deleting point: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files or 'image_id' not in request.form:
        return jsonify({"error": "No file or image_id provided"}), 400
    
    file = request.files['file']
    image_filename = request.form['image_id'] 
    print(image_filename) 

    if image_filename.endswith('.img'):
        image_filename = image_filename.replace('.img', '.png')

    visualized_image = VisualizedImages.query.filter_by(visualized_filename=image_filename).first()

    if not visualized_image:
        return jsonify({"error": "Visualized image not found"}), 404
    
    image_id = visualized_image.id  

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_CSV_FOLDER'], filename)
        file.save(file_path)
        logging.info(f"{file.filename} uploaded successfully")
        
        try:
            data = pd.read_csv(file_path, delimiter=';')

            # Convert date strings to Python date objects
            data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y').dt.date

            # Store all data in StatisticalData table
            for index, row in data.iterrows():
                point_id = row['ID']

                new_entry = StatisticalData(
                    image_id=image_id,
                    point_id=point_id,
                    y=row['X(m)'],
                    x=row['Y(m)'],
                    h=row.get('H(m)_EGM96'),
                    replicate=row.get('replicate'),
                    sub_replicate=row.get('sub_replicate'),
                    chlorophyll=row.get('chlorophyll'),
                    rice_height=row.get('Rice_Height'),
                    spectral_num=row.get('Spectral_number'),
                    digesion=row.get('Digesion'),
                    p_conc=row.get('P_conc'),
                    k_conc=row.get('K_conc'),
                    n_conc=row.get('N_conc'),
                    chlorophyll_a=row.get('Chlorophyll_a'),
                    date=row.get('date')
                )
                db.session.add(new_entry)

            db.session.commit()

            base_data = StatisticalData.query.filter_by(image_id=image_id, point_id='BASE').first()
            if not base_data:
                return jsonify({"error": "Base point not found"}), 404

            base_northing = base_data.y  # X(m)
            base_easting = base_data.x   # Y(m)

            image_width = visualized_image.width
            image_height = visualized_image.height

            # Retrieve the display size from the request (should be sent from frontend)
            display_width = int(float(request.form.get('display_width', visualized_image.width)))
            display_height = int(float(request.form.get('display_height', visualized_image.height)))


            # calculate and store pixel coordinates in the Points table
            other_points = StatisticalData.query.filter(StatisticalData.image_id == image_id, StatisticalData.point_id != 'BASE').all()

            for point_data in other_points:
                northing = point_data.y  # X(m)
                easting = point_data.x   # Y(m)
                point_id = point_data.point_id

                x_pixel, y_pixel = convert_to_pixels(
                    northing=northing,
                    easting=easting,
                    base_northing=base_northing,
                    base_easting=base_easting,
                    display_width=display_width,
                    display_height=display_height,
                    original_width=image_width,
                    original_height=image_height
                )
                # Check if the point already exists in Points table
                point_entry = Points.query.filter_by(image_id=image_id, point_id=point_id).first()
                
                if point_entry:
                    # Update existing entry
                    point_entry.x = x_pixel
                    point_entry.y = y_pixel
                else:
                    # Insert new entry
                    point_entry = Points(
                        image_id=image_id,
                        point_id=point_id,
                        x=x_pixel,
                        y=y_pixel
                    )
                    db.session.add(point_entry)

            db.session.commit()
            os.remove(file_path)
            
            return jsonify({"message": "CSV data uploaded and updated successfully."}), 200

        except Exception as e:
            logging.error(f"Error processing CSV: {e}")
            logging.error(traceback.format_exc())
            return jsonify({"error": "An error occurred while processing the CSV file."}), 500

    return jsonify({"error": "Invalid file format"}), 400

@app.route('/get_points/<image_id>', methods=['GET'])
def get_points(image_id):
    if image_id.endswith('.img'):
        image_id = image_id.replace('.img', '.png')

    visualized_image = VisualizedImages.query.filter_by(visualized_filename = image_id ).first()

    image_id = visualized_image.id

    points = Points.query.filter_by(image_id=image_id).all()
    if not points:
        return jsonify({"error": "No points found for the provided image ID"}), 404
    
    # Prepare the response data
    points_data = []
    for point in points:
        points_data.append({
            "point_id": point.point_id,
            "x": point.x,
            "y": point.y
        })
    
    return jsonify(points_data), 200

@app.route('/get_statistical_data', methods=['POST'])
def get_statistical_data():
    try:
        # Expecting a list of point IDs or coordinates from the frontend
        point_ids = request.json.get('point_ids', [])
        print(point_ids)

        if not point_ids:
            return jsonify({"error": "No point IDs provided"}), 400

        statistical_data = StatisticalData.query.filter(StatisticalData.point_id.in_(point_ids)).all()

        if not statistical_data:
            return jsonify({"error": "No statistical data found for the provided points"}), 404

        # Prepare the response data
        data_response = []
        for data in statistical_data:
            data_response.append({
                "point_id": data.point_id,
                "x": data.x,
                "y": data.y,
                "h": data.h,
                "replicate": data.replicate,
                "sub_replicate": data.sub_replicate,
                "chlorophyll": data.chlorophyll,
                "rice_height": data.rice_height,
                "spectral_num": data.spectral_num,
                "digesion": data.digesion,
                "p_conc": data.p_conc,
                "k_conc": data.k_conc,
                "n_conc": data.n_conc,
                "chlorophyll_a": data.chlorophyll_a,
                "date": data.date.isoformat() if data.date else None
            })

        return jsonify(data_response), 200

    except Exception as e:
        logging.error(f"Error retrieving statistical data: {e}")
        return jsonify({"error": "An error occurred while retrieving statistical data"}), 500
    
@app.route('/delete_data', methods=['POST'])
def delete_data():
    try:
        data = request.json

        filename = data.get('visualized_filename')

        if filename:
            image = VisualizedImages.query.filter_by(visualized_filename=filename).first()
        else:
            return jsonify({'error': 'image_id or visualized_filename is required'}), 400

        if not image:
            return jsonify({'error': 'Image not found'}), 404

        Points.query.filter_by(image_id=image.id).delete()

        StatisticalData.query.filter_by(image_id=image.id).delete()

        db.session.commit()
        return jsonify({'message': 'Data deleted successfully'}), 200
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    

@app.route('/statistical_data_table', methods=['GET'])  
def statistical_data():
    try:
        # Query to get unique point_id data, keeping the first occurrence of duplicates
        data = db.session.query(
            StatisticalData.id,
            StatisticalData.image_id,
            StatisticalData.point_id,
            StatisticalData.x,
            StatisticalData.y,
            StatisticalData.h,
            StatisticalData.replicate,
            StatisticalData.sub_replicate,
            StatisticalData.chlorophyll,
            StatisticalData.rice_height,
            StatisticalData.spectral_num,
            StatisticalData.digesion,
            StatisticalData.p_conc,
            StatisticalData.k_conc,
            StatisticalData.n_conc,
            StatisticalData.chlorophyll_a,
            StatisticalData.date
        ).group_by(
            StatisticalData.point_id, 
            StatisticalData.image_id, 
            StatisticalData.date
        ).all()

        # Convert data to list of dicts for JSON response
        result = []
        for row in data:
            result.append({
                "id": row.id,
                "image_id": row.image_id,
                "point_id": row.point_id,
                "x": row.x,
                "y": row.y,
                "h": row.h,
                "replicate": row.replicate,
                "sub_replicate": row.sub_replicate,
                "chlorophyll": row.chlorophyll,
                "rice_height": row.rice_height,
                "spectral_num": row.spectral_num,
                "digesion": row.digesion,
                "p_conc": row.p_conc,
                "k_conc": row.k_conc,
                "n_conc": row.n_conc,
                "chlorophyll_a": row.chlorophyll_a,
                "date": row.date.strftime("%Y-%m-%d")
            })

        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/n_concentration', methods=['GET'])
def get_n_concentration_data():
    try:
        # Query all data excluding point_id 'BASE'
        data = StatisticalData.query.with_entities(
            StatisticalData.point_id, StatisticalData.n_conc, StatisticalData.date).filter(
                StatisticalData.point_id != 'BASE').all()

        # Format the data as needed by Chart.js
        formatted_data = {}
        for record in data:
            point_id = record.point_id
            date = record.date.strftime("%Y-%m-%d")  # Convert date to string
            n_conc = record.n_conc

            if point_id not in formatted_data:
                formatted_data[point_id] = {"dates": [], "n_conc_values": []}

            formatted_data[point_id]["dates"].append(date)
            formatted_data[point_id]["n_conc_values"].append(n_conc)
            # print(formatted_data)

        return jsonify(formatted_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/k_concentration', methods=['GET'])
def get_k_concentration_data():
    try:
        # Query all K concentration data excluding point_id 'BASE'
        data = StatisticalData.query.with_entities(
            StatisticalData.point_id, StatisticalData.k_conc, StatisticalData.date).filter(
                StatisticalData.point_id != 'BASE').all()

        # Format the data
        formatted_data = {}
        for record in data:
            point_id = record.point_id
            date = record.date.strftime("%Y-%m-%d")  # Convert date to string
            k_conc = record.k_conc

            if point_id not in formatted_data:
                formatted_data[point_id] = {"dates": [], "k_conc_values": []}

            formatted_data[point_id]["dates"].append(date)
            formatted_data[point_id]["k_conc_values"].append(k_conc)

        return jsonify(formatted_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/p_concentration', methods=['GET'])
def get_p_concentration_data():
    try:
        # Query all P concentration data excluding point_id 'BASE'
        data = StatisticalData.query.with_entities(
            StatisticalData.point_id, StatisticalData.p_conc, StatisticalData.date).filter(
                StatisticalData.point_id != 'BASE').all()

        # Format the data
        formatted_data = {}
        for record in data:
            point_id = record.point_id
            date = record.date.strftime("%Y-%m-%d")  # Convert date to string
            p_conc = record.p_conc

            if point_id not in formatted_data:
                formatted_data[point_id] = {"dates": [], "p_conc_values": []}

            formatted_data[point_id]["dates"].append(date)
            formatted_data[point_id]["p_conc_values"].append(p_conc)

        return jsonify(formatted_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chlorophyll_a', methods=['GET'])
def get_chlorophyll_a_data():
    try:
        # Query all Chlorophyll A data excluding point_id 'BASE'
        data = StatisticalData.query.with_entities(StatisticalData.point_id, StatisticalData.chlorophyll_a, StatisticalData.date).filter(StatisticalData.point_id != 'BASE').all()

        # Format the data
        formatted_data = {}
        for record in data:
            point_id = record.point_id
            date = record.date.strftime("%Y-%m-%d")  # Convert date to string
            chlorophyll_a = record.chlorophyll_a

            if point_id not in formatted_data:
                formatted_data[point_id] = {"dates": [], "chlorophyll_a_values": []}

            formatted_data[point_id]["dates"].append(date)
            formatted_data[point_id]["chlorophyll_a_values"].append(chlorophyll_a)

        return jsonify(formatted_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for file upload and spectral graph generation
@app.route('/upload_reflectance_data', methods=['POST'])
def upload_reflectance_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file temporarily
    filepath = os.path.join(app.config['UPLOAD_REFLECTANCE_DATA'], file.filename)
    file.save(filepath)

    try:
        s = specdal.Spectrum(filepath=filepath)
        # Plot the graph directly using specdal's plot functionality
        plt.figure()
        s.plot()
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title('Spectral Reflectance')

        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Remove the temporary file
        os.remove(filepath)

        # Return the image as a response
        return send_file(img, mimetype='image/png')

    except Exception as e:
        return f"Error processing file: {str(e)}", 400
        
if __name__ == "__main__":
    app.run(debug=True)

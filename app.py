import glob

import face_recognition
from flask import send_from_directory
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from ALPR import *
from currentLocation import *
from dbFunc import *
from dbFunctions import find_lp_owner

# Initialize app
app = Flask(__name__, static_url_path='', )

# Code needed for image uploading
app.config['SECRET_KEY'] = 'capstone123'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images are allowed.'),
            FileRequired('File field should not be empty.')
        ]
    )
    submit = SubmitField('Upload')


currentName = ""
percent_accuracy = None
display_lpResult = ""
display_oResult = ""
display_iResult = ""
plate = ""

### NOTE: All images must have the following format to loaded and read properly:  'name.jpg' ####

# Create arrays of known face encodings and their names
known_face_names = list()
known_face_encodings = list()

# Load all images names into lists 

images = (os.listdir('images/'))
imagesRGB = [face_recognition.load_image_file(file) for file in glob.glob('images/*.jpg')]

# Add the known names as strings into an array & process their encodings
for i in range(len(images)):
    known_face_names.append(images[i][:-4])
    sample_face_encoding = face_recognition.face_encodings(imagesRGB[i])[0]
    known_face_encodings.append(sample_face_encoding)

process_this_frame = True


def gen_frames_lp():
    global plate
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                # cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

                # image = cv2.imread("img2.jpeg")
                # image = cv2.resize(image, (735, 417))
                frame_resized = cv2.resize(frame, (620, 480))  # image rescaling



                # # convert to grey scale (black and white)
                # gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                # gray = cv2.bilateralFilter(gray, 11, 17, 17)  # remove blurring
#
                # edged = cv2.Canny(gray, 100, 200)  # edge detection
#
                # binary = cv2.bitwise_not(gray)
                # contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # contours = imutils.grab_contours(contours)
                # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                # # cnt = contours[0]
                # for c in contours:  # approximating contour
                #     peri = cv2.arcLength(c, True)
                #     # # if contour has 4 points, then we have found our screen
                #     approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                #     if 4 <= len(approx) <= 4:
                #         cnt = approx
                #         break
#
                # mask = np.zeros(gray.shape, np.uint8)  # masking image excluding plate
                # image_2 = cv2.drawContours(mask, contours, 0, 255, -1)
                # # image_2 = cv2.bitwise_and(frame, frame, mask=mask)

                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)  # gray image
                noise_reduction = cv2.bilateralFilter(gray, 13, 15, 15)  # apply filters for noise reduction
                edged = cv2.Canny(noise_reduction, 30, 200)  # detect edges

                points = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # approximate contours
                contours = imutils.grab_contours(points)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # find top 10 contours

                # loop top 10 contours to find license plate
                cnt = None
                for contour in contours:
                    approx = cv2.approxPolyDP(contour, 10, True)
                    if len(approx) == 4:
                        cnt = approx
                        break

                mask = np.zeros(gray.shape, np.uint8)  # mask image
                new_image = cv2.drawContours(mask, [cnt], 0, 255, -1)
                new_image = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)



                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                cropped_image = gray[topx:bottomx + 1, topy:bottomy + 1]

                text = pytesseract.image_to_string(cropped_image, config='--psm 7')
                text = text.lower()
                text = text.replace(" ", "")
                try:
                    t = re.search(r"([a-z]{1,4}[0-9]{1,6})", text).group(0)
                    t = t.replace("ontario", "", 1)
                    t = t.replace("to", "", 1)
                    t = t.replace("discover", "", 1)
                    t = t.replace("mar", "", 1)

                    if len(t) > 2:
                        print("License plate number is: ", t)
                        plate = t

                except Exception as e:
                    pass

                ret, buffer = cv2.imencode('.jpg', cropped_image)
                frame_bytes = buffer.tobytes()
            except Exception as e:
                print(e)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def gen_frames_face(debug=False, filename=None):
    """Generates facial predictions either from camera or local files"""
    if not debug:
        camera = cv2.VideoCapture(0)

    while True:
        if debug:
            frame = cv2.imread(filename)
            success = True
        else:
            success, frame = camera.read()
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                ## Calculate the accuracy of the face detected (compared with the highest matched face)
                global percent_accuracy
                percent_accuracy = np.round((1 - face_distances[best_match_index]) * 100, 2)
                if matches[best_match_index] and percent_accuracy >= 50:
                    name = known_face_names[best_match_index]

                face_names.append(name)  ## Label of the image being matched!

                ## Only print accuracy if it redetects a new person/unknown
                global currentName
                if ((name != currentName) or debug):
                    print("Accuracy: " + str(percent_accuracy) + "% " + name)

                currentName = name

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/plate')
def plate_poll():
    global plate
    """send current content"""
    return plate


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/lpPage', methods=['GET', 'POST'])
def lpPage():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('lpPage.html', displayName=currentName, displayAccuracy=str(percent_accuracy) + " %")


@app.route('/fPage', methods=['GET', 'POST'])
def fPage():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('fPage.html', displayName=currentName, displayAccuracy=str(percent_accuracy) + " %")


@app.route('/lpPageFile', methods=['GET', 'POST'])
def fPageFile():
    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template('fPageFile.html', displayName=currentName, displayAccuracy=str(percent_accuracy) + " %")


@app.route('/lpPageFile', methods=['GET', 'POST'])
def lpPageFile():
    if request.method == 'POST':
        return redirect(url_for('index'))

    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        global display_lpResult
        display_lpResult = readLP(filename)

        # Initialize database connection
        con = sqlite3.Connection('Database.db')
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # dbQuery = None if the plate isnt in the database
        dbQuery = find_lp_owner(display_lpResult, cur)
        print(dbQuery)
        print(type(dbQuery))
        global display_oResult
        global display_iResult
        if dbQuery is None:
            display_oResult = "Not Found"
            display_iResult = "Not Found"
        else:
            display_oResult = dbQuery['Owner']
            display_iResult = dbQuery['Info']

        # Close connection to database
        con.close()

        print("Uploaded file: " + filename)  ## Variable 'filename' stores the name of the image selected, e.g. im4.png
    else:
        file_url = None
    return render_template('lpPageFile.html',
                           displayGpsResult=displayL(),
                           form=form,
                           file_url=file_url,
                           display_lpResult=display_lpResult,
                           display_oResult=display_oResult,
                           display_iResult=display_iResult
                           )


@app.route('/video_feed_lp')
def video_feed_face():
    return Response(gen_frames_face(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_face')
def video_feed_lp():
    return Response(gen_frames_lp(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


@app.route("/currentName")
def updateCurrentName():
    return f"{currentName}"


@app.route("/displayAccuracy")
def updateAccuracy():
    return str(percent_accuracy) + "%"


@app.route("/displayLocation")
def displayLocation():
    displayGpsResult = str(getLocation())
    return displayGpsResult


def test1():
    """All results are unknown means very low false positives, what is good result"""
    for i in range(100):
        gen = gen_frames_face(debug=True, filename="test/face_image_" + str(i) + ".jpg")

        print(i)
        gen.__next__()

        gen.close()


def test2():
    """Shows only recognizable faces"""
    for i in ("Ahmad", "Humza", "Leonardo", "Mohamad",):
        gen = gen_frames_face(debug=True, filename="images/" + i + ".jpg")

        print(i)
        gen.__next__()

        gen.close()


if __name__ == '__main__':
    # test1()
    # test2()
    app.run(host='0.0.0.0', debug=True, threaded=True)

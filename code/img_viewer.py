import PySimpleGUI as sg
# import PySimpleGUIQt as sg
import os.path
import PIL.Image
import io
import base64

def convert_to_bytes(file_or_bytes, resize=None):

    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.LANCZOS)
    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()


# --------------------------------- Define Layout ---------------------------------

# First the window layout...2 columns

left_col = [[sg.Text('Folder'), sg.In(size=(25, 1), enable_events=True, key='-FOLDER-'), sg.FolderBrowse()],
            [sg.Listbox(values=[], enable_events=True, size=(40, 20), key='-FILE LIST-')],
            [sg.Button('Evaluate Image', key='evaluate')]]

# For now will only show the name of the file that was chosen
images_col = [[sg.Text('You choose from the list:')],
              [sg.Text(size=(40, 1), key='-TOUT-')],
              [sg.Image(key='-IMAGE-')]]

results_col = [[sg.Text("Analysis Results", key="results_text")],
              [sg.Text("Microhemorrhages:", key="microhemorrhages_text")],
              [sg.Text("Giant Capillaries:", key="giant_capillaries_text")],
              [sg.Text("Normal Density:", key="normal_text")],
              [sg.Text("Bushy Capillaries:", key="bushy_text")]]

# ----- Full layout -----
layout = [
    [sg.Column(left_col, element_justification='c'), sg.VSeperator(), sg.Column(images_col, element_justification='c'),
     sg.VSeperator(), sg.Column(results_col, element_justification='c')]]
# --------------------------------- Create Window ---------------------------------
window = sg.Window('Multiple Format Image Viewer', layout, resizable=True)

# ----- Run the Event Loop -----
# --------------------------------- Event Loop ---------------------------------
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == '-FOLDER-':  # Folder name was filled in, make a list of files in the folder
        folder = values['-FOLDER-']
        try:
            file_list = os.listdir(folder)  # get list of files in folder
        except:
            file_list = []
        fnames = [f for f in file_list if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", "jpeg", ".tiff", ".bmp"))]
        window['-FILE LIST-'].update(fnames)
    elif event == '-FILE LIST-':  # A file was chosen from the listbox
        try:
            filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            window['-TOUT-'].update(filename)
            window['-IMAGE-'].update(data=convert_to_bytes(filename, resize=(768, 768)))
        except Exception as E:
            print(f'** Error {E} **')
            pass  # something weird happened making the full filename
    if event == 'evaluate':
        results = ["test", "test", "test", "test"]
        window['microhemorrhages_text'].update(results[0])
        window['giant_capillaries_text'].update(results[1])
        window['normal_text'].update(results[2])
        window['bushy_text'].update(results[3])
# --------------------------------- Close & Exit ---------------------------------
window.close()
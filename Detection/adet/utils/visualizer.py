import numpy as np
from detectron2.utils.visualizer import Visualizer
import cv2
import matplotlib.pyplot as plt
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import os

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"


def make_groups():
    # dictionary = 'aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ'
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()

TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D‑", "d‑"]


def correct_tone_position(word):
    word = word[:-1]
    if len(word) < 2:
        pass
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char


def decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    if len(recognition) < 1:
        return recognition
    if recognition[-1] in TONES:
        if len(recognition) < 2:
            return recognition
        replace_char = correct_tone_position(recognition)
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition

class TextVisualizer(Visualizer):
    def draw_instance_predictions(self,op, predictions):
        beziers = predictions.beziers.numpy()
        scores = predictions.scores.tolist()
        recs = predictions.recs

        self.overlay_instances(beziers, recs, scores, op)

        return self.output

    def _bezier_to_poly(self, bezier):
        # bezier to polygon
        u = np.linspace(0, 1, 20)
        bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
        points = (
            np.outer((1 - u) ** 3, bezier[:, 0])
            + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1])
            + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2])
            + np.outer(u ** 3, bezier[:, 3])
        )
        points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)

        return points

    def _decode_recognition(self, rec):
        #        CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        # CTLABELS = ['^', '\\', '}', 'ỵ', '>', '<', '{', '~', '`', '°', '$', 'ẽ', 'ỷ', 'ẳ', '_', 'ỡ', ';', '=', 'Ẳ', 'j', '[', ']', 'ẵ', '?', 'ẫ', 'Ẵ', 'ỳ', 'Ỡ', 'ẹ', 'è', 'z', 'ỹ', 'ằ', 'õ', 'ũ', 'Ẽ', 'ỗ', 'ỏ', '@', 'Ằ', 'Ỳ', 'Ẫ', 'ù', 'ử', '#', 'Ẹ', 'Z', 'Õ', 'ĩ', 'Ỏ', 'È', 'Ỷ', 'ý', 'Ũ', '*', 'ò', 'é', 'q', 'ở', 'ổ', 'ủ', 'ẩ', 'ã', 'ẻ', 'J', 'ữ', 'ễ', 'ặ', '+', 'ứ', 'Ỹ', 'ự', 'ụ', 'Ỗ', '%', 'ắ', 'ồ', '"', 'ề', 'ể', 'ỉ', 'ợ', '!', 'Ẻ', 'ừ', 'ọ', '&', 'ì', 'É', 'ậ', 'Ù', 'Ặ', 'x', 'Ỉ', 'ú', 'í', 'ó', 'Ẩ', 'ị', 'ế', 'Ứ', 'â', 'ấ', 'ầ', 'ớ', 'ă', 'Ủ', 'Ĩ', '(', 'Ắ', 'Ừ', ')', 'ờ', 'Ý', 'Ễ', 'Ã', 'ô', 'ộ', 'Ữ', 'Ợ', 'ả', 'Ở', 'ệ', 'W', 'ơ', 'Ổ', 'ố', 'Ề', 'f', 'Ử', 'ạ', 'w', 'Ò', 'Ự', 'Ụ', 'Ú', 'Ồ', 'ê', 'Ó', 'Ì', 'b', 'Í', 'Ể', 'đ', 'Ớ', '/', 'k', 'Ă', 'v', 'Ị', 'Ậ', 'Ọ', 'd', 'Ầ', 'Ấ', 'ư', 'á', 'Ế', ' ', 'p', 'Ơ', 'F', 'Ả', 'Ộ', 'Ê', 'Ờ', 's', '-', 'à', 'y', 'Ố', 'l', 'Â', 'Q', ',', 'X', 'Ệ', 'Ạ', 'Ô', 'r', ':', '6', '7', 'u', '4', 'm', '5', 'e', '8', 'c', 'Ư', 'Á', '9', 'D', '3', 'o', '.', 'Y', 'g', 'K', 'a', 'À', 't', '2', 'B', 'E', 'V', 'R', '1', 'S', 'i', 'L', 'P', 'Đ', 'h', 'U', '0', 'M', 'O', 'n', 'A', 'G', 'I', 'C', 'T', 'H', 'N']
        CTLABELS = [
            " ",
            "!",
            '"',
            "#",
            "$",
            "%",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ":",
            ";",
            "<",
            "=",
            ">",
            "?",
            "@",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "[",
            "\\",
            "]",
            "^",
            "_",
            "`",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "{",
            "|",
            "}",
            "~",
            "ˋ",
            "ˊ",
            "﹒",
            "ˀ",
            "˜",
            "ˇ",
            "ˆ",
            "˒",
            "‑",
        ]

        s = ""
        for c in rec:
            c = int(c)
            if c < 104:
                s += CTLABELS[c]
            elif c == 104:
                s += u"口"
        return decoder(s)
    
    def init(self , s , f , cnt , alpha = 0.2):
      try:
        #print("test1: " , "./adet/data/datasets/train/" + s + ".txt")
        r = open("./adet/data/datasets/train/" + s + '.txt')
        for line in r:
          #print(line)
          L = line.split(",")
          #print("L: " , L)
          polygon = [[L[0] , L[1]] , [L[12] , L[13]] , [L[14] , L[15]] , [L[26] , L[27]]]
          #print("polygon: " , polygon)
          self.draw_polygon(polygon, "blue", alpha=alpha)
          # draw text in the top left corner
          b = str(int(polygon[0][0])) + "," + str(int(polygon[0][1])) + "," + str(int(polygon[1][0])) + "," + str(int(polygon[1][1])) + "," + str(int(polygon[2][0])) + "," + str(int(polygon[2][1])) + "," + str(int(polygon[3][0])) + "," + str(int(polygon[3][1]))
          strr = str(int(polygon[0][0])) + "," + str(int(polygon[0][1])) + "," + str(int(polygon[1][0])) + "," + str(int(polygon[1][1])) + "," + str(int(polygon[2][0])) + "," + str(int(polygon[2][1])) + "," + str(int(polygon[3][0])) + "," + str(int(polygon[3][1])) + "," + "###"
          f.writelines(strr)
          f.write('\n')
          print("bounding box " + str(cnt) + ": " + b)
          cnt += 1

        #print("test2")
      except:
        pass

    def _ctc_decode_recognition(self, rec):
        # CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        # CTLABELS = ['^', '\\', '}', 'ỵ', '>', '<', '{', '~', '`', '°', '$', 'ẽ', 'ỷ', 'ẳ', '_', 'ỡ', ';', '=', 'Ẳ', 'j', '[', ']', 'ẵ', '?', 'ẫ', 'Ẵ', 'ỳ', 'Ỡ', 'ẹ', 'è', 'z', 'ỹ', 'ằ', 'õ', 'ũ', 'Ẽ', 'ỗ', 'ỏ', '@', 'Ằ', 'Ỳ', 'Ẫ', 'ù', 'ử', '#', 'Ẹ', 'Z', 'Õ', 'ĩ', 'Ỏ', 'È', 'Ỷ', 'ý', 'Ũ', '*', 'ò', 'é', 'q', 'ở', 'ổ', 'ủ', 'ẩ', 'ã', 'ẻ', 'J', 'ữ', 'ễ', 'ặ', '+', 'ứ', 'Ỹ', 'ự', 'ụ', 'Ỗ', '%', 'ắ', 'ồ', '"', 'ề', 'ể', 'ỉ', 'ợ', '!', 'Ẻ', 'ừ', 'ọ', '&', 'ì', 'É', 'ậ', 'Ù', 'Ặ', 'x', 'Ỉ', 'ú', 'í', 'ó', 'Ẩ', 'ị', 'ế', 'Ứ', 'â', 'ấ', 'ầ', 'ớ', 'ă', 'Ủ', 'Ĩ', '(', 'Ắ', 'Ừ', ')', 'ờ', 'Ý', 'Ễ', 'Ã', 'ô', 'ộ', 'Ữ', 'Ợ', 'ả', 'Ở', 'ệ', 'W', 'ơ', 'Ổ', 'ố', 'Ề', 'f', 'Ử', 'ạ', 'w', 'Ò', 'Ự', 'Ụ', 'Ú', 'Ồ', 'ê', 'Ó', 'Ì', 'b', 'Í', 'Ể', 'đ', 'Ớ', '/', 'k', 'Ă', 'v', 'Ị', 'Ậ', 'Ọ', 'd', 'Ầ', 'Ấ', 'ư', 'á', 'Ế', ' ', 'p', 'Ơ', 'F', 'Ả', 'Ộ', 'Ê', 'Ờ', 's', '-', 'à', 'y', 'Ố', 'l', 'Â', 'Q', ',', 'X', 'Ệ', 'Ạ', 'Ô', 'r', ':', '6', '7', 'u', '4', 'm', '5', 'e', '8', 'c', 'Ư', 'Á', '9', 'D', '3', 'o', '.', 'Y', 'g', 'K', 'a', 'À', 't', '2', 'B', 'E', 'V', 'R', '1', 'S', 'i', 'L', 'P', 'Đ', 'h', 'U', '0', 'M', 'O', 'n', 'A', 'G', 'I', 'C', 'T', 'H', 'N']
        CTLABELS = [
            " ",
            "!",
            '"',
            "#",
            "$",
            "%",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ":",
            ";",
            "<",
            "=",
            ">",
            "?",
            "@",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "[",
            "\\",
            "]",
            "^",
            "_",
            "`",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "{",
            "|",
            "}",
            "~",
            "ˋ",
            "ˊ",
            "﹒",
            "ˀ",
            "˜",
            "ˇ",
            "ˆ",
            "˒",
            "‑",
        ]

        # ctc decoding
        last_char = False
        s = ""
        for c in rec:
            c = int(c)
            if c < 104:
                if last_char != c:
                    s += CTLABELS[c]
                    last_char = c
            elif c == 104:
                s += u"口"
            else:
                last_char = False
        return decoder(s)

    def overlay_instances(self, beziers, recs, scores, op, alpha=0.2):
        cnt = 1
        color = (0.1, 0.2, 0.5)
        # color = 'green'
        ok = 0
        s = ""
        for i in op:
          if i == "/":
            if ok >= 2:
                s = s + i
            ok += 1
            
          elif i == '.':
            ok = 0
          elif ok >= 2:
            s = s + i
        tail = op.split(".")[-1]
        #print()
        #print("name:" , s)
        #print("tail: " , tail)
        direct = "./data/submission_output/" + s + "." + tail + ".txt"
        config = Cfg.load_config_from_file('./recog_config.yml')
        config['weights'] = './save_models/recog.pth' 
        config['device'] = 'cuda:0'
        detector = Predictor(config)
        print("direct:" , direct)
        with open(direct, 'w') as f:
          for bezier, rec, score in zip(beziers, recs, scores):
              polygon = self._bezier_to_poly(bezier)
              self.draw_polygon(polygon, "blue", alpha=alpha)
              b = str(int(polygon[0][0])) + "," + str(int(polygon[0][1])) + "," + str(int(polygon[19][0])) + "," + str(int(polygon[19][1])) + "," + str(int(polygon[20][0])) + "," + str(int(polygon[20][1])) + "," + str(int(polygon[39][0])) + "," + str(int(polygon[39][1]))
              #print("bounding box " + str(cnt) + ": " + b)
              # draw text in the top left corner
              text = self._decode_recognition(rec)		
              cnt += 1
              #text = "{:.3f}: {}".format(score, text)
              lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
              text_pos = polygon[0]
              horiz_align = "left"
              font_size = self._default_font_size
              img = cv2.imread(op)
              L = [0] * 9
              L[0] = int(polygon[0][0])
              L[1] = int(polygon[0][1])
              L[2] = int(polygon[19][0])
              L[3] = int(polygon[19][1])
              L[4] = int(polygon[20][0])
              L[5] = int(polygon[20][1])
              L[6] = int(polygon[39][0])
              L[7] = int(polygon[39][1])
              x = int(min(L[0] , L[2] , L[4] , L[6]))
              w = int(max(L[0] , L[2] , L[4] , L[6]))
              y = int(min(L[1] , L[3] , L[5] , L[7]))
              h = int(max(L[1] , L[3] , L[5] , L[7]))
              pts = np.array([[L[0] , L[1]] , [L[2] , L[3]] , [L[4] , L[5]] , [L[6] , L[7]]])
              mask = np.zeros(img.shape[:2], np.uint8)
              cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
              dst = cv2.bitwise_and(img , img, mask=mask)
              #plt.imshow(dst[y:h , x:w])
              color_coverted = cv2.cvtColor(dst[y:h , x:w], cv2.COLOR_BGR2RGB)
              pil_image=Image.fromarray(color_coverted)
              #plt.imshow(pil_image)
              #plt.show()
              # dự đoán 
              guess = detector.predict(pil_image, return_prob=True) 
              guess = list(guess)
              if (guess[1]<0.7 or text == '#' or text == '##' or text == '###'):
                 guess[0]='###'
              if guess[1]<0.7 and dst[y:h , x:w].shape[0]>2*dst[y:h , x:w].shape[1]:
                  rotated_image1 = pil_image.rotate(270 , expand = 1)
                  rotated_image2 = pil_image.rotate(90 , expand = 1)
                  s1=detector.predict(rotated_image1, return_prob=True)
                  s2=detector.predict(rotated_image2, return_prob=True)
                  if s1[1]>0.7 and s1[1]>s2[1]:
                     guess[0]=s1[0]
                     cv2.imwrite('./picCut/' + s + '_' + str(cnt) + ".png", dst[y:h , x:w])
                  elif s2[1]>0.7 and s2[1]>s1[1]:
                     guess[0]=s2[0]
                     cv2.imwrite('./picCut/' + s + '_' + str(cnt) + ".png", dst[y:h , x:w])
              if("#" not in guess[0]):
                strr = str(int(polygon[0][0])) + "," + str(int(polygon[0][1])) + "," + str(int(polygon[19][0])) + "," + str(int(polygon[19][1])) + "," + str(int(polygon[20][0])) + "," + str(int(polygon[20][1])) + "," + str(int(polygon[39][0])) + "," + str(int(polygon[39][1])) + "," + guess[0]
                f.writelines(strr)
                f.write('\n')
              #print("model1: " , guess)
              #print("model2: " , text)
              #self.draw_text(
              #      text,
              #      text_pos,
              #      color='yellow',
              #      horizontal_alignment=horiz_align,
              #      font_size=font_size,
              #  )
          f.close()

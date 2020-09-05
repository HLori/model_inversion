import numpy as np
import os
import torch
from torch import nn
from torchvision import transforms
from torch import optim
from ResNet import ResNet18
from PIL import Image
import random


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
BINARY_SEARCH_STEPS = 1  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 101  # number of iterations to perform gradient descent
# ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 0.01  # larger values converge faster to less accurate results
INITIAL_CONST = 1  # the initial constant lambda to pick as a first guess
IMG_SIZE = 224
CHANNELS = 3
NUM_LABEL = 200
name2idx = {'Aaron_Eckhart': 0, 'Adam_Brody': 1, 'Adam_McKay': 2, 'Adam_Sandler': 3, 'Adrien_Brody': 4, 'Al_Pacino': 5, 'Alan_Alda': 6, 'Alan_Arkin': 7, 'Alan_Rickman': 8, 'Alec_Baldwin': 9, 'Alexander_Skarsgård': 10, 'Alfred_Molina': 11, 'Amaury_Nolasco': 12, 'Andy_Garcia': 13, 'Andy_Richter': 14, 'Andy_Serkis': 15, 'Anthony_Hopkins': 16, 'Anthony_Stewart_Head': 17, 'Antonio_Banderas': 18, 'Arnold_Schwarzenegger': 19, 'Arnold_Vosloo': 20, 'Ashton_Kutcher': 21, 'Ben_Affleck': 22, 'Ben_Kingsley': 23, 'Ben_McKenzie': 24, 'Ben_Stiller': 25, 'Benicio_Del_Toro': 26, 'Bernard_Hill': 27, 'Bernie_Mac': 28, 'Bill_Cosby': 29, 'Bill_Hader': 30, 'Bill_Murray': 31, 'Bill_Nighy': 32, 'Billy_Bob_Thornton': 33, 'Billy_Boyd': 34, 'Billy_Burke': 35, 'Billy_Zane': 36, 'Brad_Garrett': 37, 'Brad_Pitt': 38, 'Bradley_Cooper': 39, 'Brendan_Fraser': 40, 'Bruce_Greenwood': 41, 'Bruce_Willis': 42, 'Burt_Reynolds': 43, 'Cam_Gigandet': 44, 'Cary_Elwes': 45, 'Casey_Affleck': 46, 'Channing_Tatum': 47, 'Charlie_Day': 48, 'Charlie_Sheen': 49, 'Chazz_Palminteri': 50, 'Chris_Evans': 51, 'Chris_Kattan': 52, 'Chris_Noth': 53, 'Chris_Rock': 54, 'Christian_Bale': 55, 'Christian_Slater': 56, 'Christopher_Lloyd': 57, 'Christopher_Reeve': 58, 'Clint_Eastwood': 59, 'Clive_Owen': 60, 'Colin_Farrell': 61, 'Colin_Firth': 62, 'Colin_Hanks': 63, 'Dan_Lauria': 64, 'Daniel_Craig': 65, 'Daniel_Day-Lewis': 66, 'Daniel_Radcliffe': 67, 'Danny_Glover': 68, 'Danny_Masterson': 69, 'Danny_Trejo': 70, 'David_Boreanaz': 71, 'David_Cross': 72, 'David_Duchovny': 73, 'David_Schwimmer': 74, 'David_Wenham': 75, 'Dean_Cain': 76, 'Denzel_Washington': 77, 'Dermot_Mulroney': 78, 'Desmond_Harrington': 79, 'Diego_Luna': 80, 'Dominic_Monaghan': 81, 'Don_Cheadle': 82, 'Dustin_Hoffman': 83, 'Dwayne_Johnson': 84, 'Ed_Harris': 85, 'Edi_Gathegi': 86, 'Elijah_Wood': 87, 'Emile_Hirsch': 88, 'Eric_Dane': 89, 'Ethan_Hawke': 90, 'Ewan_McGregor': 91, 'Freddy_Prinze_Jr.': 92, 'Freddy_Rodríguez': 93, 'Gabriel_Macht': 94, 'Gary_Dourdan': 95, 'Gary_Oldman': 96, 'Gene_Hackman': 97, 'Geoffrey_Rush': 98, 'George_Clooney': 99, 'George_Lopez': 100, 'Gerard_Butler': 101, 'Giovanni_Ribisi': 102, 'Hal_Holbrook': 103, 'Hank_Azaria': 104, 'Harrison_Ford': 105, 'Harry_Connick_Jr.': 106, 'Hayden_Christensen': 107, 'Heath_Ledger': 108, 'Hector_Elizondo': 109, 'Hugh_Grant': 110, 'Hugh_Jackman': 111, 'Hugo_Weaving': 112, 'Ian_Holm': 113, 'Ian_McKellen': 114, 'Ioan_Gruffudd': 115, 'J.K._Simmons': 116, 'Jack_Nicholson': 117, 'Jackie_Chan': 118, 'Jackson_Rathbone': 119, 'Jaden_Smith': 120, 'Jake_Gyllenhaal': 121, 'Jake_Weber': 122, 'James_Brolin': 123, 'James_Frain': 124, 'James_Franco': 125, 'James_Marsden': 126, 'James_McAvoy': 127, 'James_Remar': 128, 'Jamie_Foxx': 129, 'Jared_Padalecki': 130, 'Jason_Bateman': 131, 'Jason_Behr': 132, 'Jason_Biggs': 133, 'Jason_Lee': 134, 'Jason_Statham': 135, 'Jason_Sudeikis': 136, 'Jay_Baruchel': 137, 'Jean-Claude_Van_Damme': 138, 'Jean_Reno': 139, 'Jeffrey_Tambor': 140, 'Jensen_Ackles': 141, 'Jeremy_Irons': 142, 'Jeremy_Sisto': 143, 'Jerry_Seinfeld': 144, 'Jesse_Eisenberg': 145, 'Jet_Li': 146, 'Jim_Beaver': 147, 'Jim_Carrey': 148, 'Jim_Caviezel': 149, 'Jimmy_Fallon': 150, 'Joaquin_Phoenix': 151, 'Joe_Manganiello': 152, 'Joe_Pantoliano': 153, 'John_Cleese': 154, 'John_Cusack': 155, 'John_Krasinski': 156, 'John_Malkovich': 157, 'John_Noble': 158, 'John_Travolta': 159, 'Johnny_Depp': 160, 'Jon_Hamm': 161, 'Jon_Voight': 162, 'Jonah_Hill': 163, 'Jonathan_Rhys_Meyers': 164, 'Jonathan_Sadowski': 165, 'Josh_Brolin': 166, 'Josh_Duhamel': 167, 'Josh_Hartnett': 168, 'Joshua_Jackson': 169, 'Jude_Law': 170, 'Justin_Long': 171, 'Justin_Timberlake': 172, 'Kal_Penn': 173, 'Karl_Urban': 174, 'Keanu_Reeves': 175, 'Kellan_Lutz': 176, 'Ken_Watanabe': 177, 'Kerr_Smith': 178, 'Kevin_Bacon': 179, 'Kevin_Connolly': 180, 'Kevin_Costner': 181, 'Kevin_McKidd': 182, 'Kiefer_Sutherland': 183, 'Kit_Harington': 184, 'Kris_Kristofferson': 185, 'Laurence_Fishburne': 186, 'Leonardo_DiCaprio': 187, 'Leslie_Neilsen': 188, 'Liev_Schreiber': 189, 'Luke_Wilson': 190, 'Mark_Ruffalo': 191, 'Mark_Wahlberg': 192, 'Martin_Henderson': 193, 'Martin_Lawrence': 194, 'Martin_Sheen': 195, 'Matthew_Broderick': 196, 'Matthew_Gray_Gubler': 197, 'Matthew_Lillard': 198, 'Matthew_McConaughey': 199}
idx2name = {v: k for k, v in name2idx.items()}


def per_image_perturbation(data_path, dest_path, dest2, model_path, device, target, learning_rate=LEARNING_RATE, max_iteration=MAX_ITERATIONS):
    # load model
    model = ResNet18()
    checkpoint = torch.load(model_path)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dice'])
    loss = nn.CrossEntropyLoss()
    target_class = torch.Tensor([target]).to(device)
    target_class = target_class.type(torch.long)
    target_pert = os.path.join(dest2, '{0}_{1}'.format(target, idx2name[target]))
    target_exam = os.path.join(dest_path, '{0}_{1}'.format(target, idx2name[target]))

    transform = transforms.Compose([
        # transforms.Normalize(mean=[0.59645426, 0.45772487, 0.39632604], std=[0.27194658, 0.23214313, 0.2273719]),
        transforms.ToTensor()
    ])

    for person in os.listdir(data_path):
        person_dir = os.path.join(data_path, person)
        person_img_list = os.listdir(person_dir)
        random.shuffle(person_img_list)
        target_path_pert = os.path.join(target_pert, person)
        target_path_exam = os.path.join(target_exam, person)
        if not os.path.exists(target_path_pert):
            os.makedirs(target_path_pert)
        if not os.path.exists(target_path_exam):
            os.makedirs(target_path_exam)

        for i in range(min(len(person_img_list), 30)):
            img = Image.open(os.path.join(person_dir, person_img_list[i]))
            img = transform(img).to(device)
            img = torch.unsqueeze(img, 0)
            delta = torch.ones_like(img, requires_grad=True)
            opt = optim.Adam([delta], lr=learning_rate)

            for step in range(max_iteration):
                # delta = torch.clamp(delta, min=0, max=255)
                output = model(delta + img)
                l = loss(output, target_class)
                opt.zero_grad()
                l.backward(retain_graph=True)
                opt.step()

                if torch.max(output) == output[0, target]:
                    new_img = transforms.ToPILImage()(torch.squeeze(delta+img, 0).cpu().detach()).convert('RGB')
                    new_img2 = transforms.ToPILImage()(torch.squeeze(delta, 0).cpu().detach()).convert('RGB')
                    new_img.save(os.path.join(target_path_exam, "{0}_from_{1}.jpg".format(i, person)))
                    new_img2.save(os.path.join(target_path_pert, "{0}_from_{1}.jpg".format(i, person)))
                    # new_img.save(os.path.join(person_dest, "{0}-{1}".format(target, person_img_list[i])))
                    # new_img2.save(os.path.join(person_dest2, "{0}-{1}".format(target, person_img_list[i])))
                    break


for i in range(100):
    per_image_perturbation('./data2_facescrub', 'adv_example', 'adv_pert', 'model/face_weights.pt', torch.device("cuda:0"), i)
    print('----------------------------------------done {0}'.format(i))
import torch
import torchvision
from torch.utils.data import DataLoader
from ResNet import CGAN
from torchvision import datasets
from torchvision import transforms
import os
from PIL import Image

name2idx = {'Aaron_Eckhart': 0, 'Adam_Brody': 1, 'Adam_McKay': 2, 'Adam_Sandler': 3, 'Adrien_Brody': 4, 'Al_Pacino': 5, 'Alan_Alda': 6, 'Alan_Arkin': 7, 'Alan_Rickman': 8, 'Alec_Baldwin': 9, 'Alexander_Skarsgård': 10, 'Alfred_Molina': 11, 'Amaury_Nolasco': 12, 'Andy_Garcia': 13, 'Andy_Richter': 14, 'Andy_Serkis': 15, 'Anthony_Hopkins': 16, 'Anthony_Stewart_Head': 17, 'Antonio_Banderas': 18, 'Arnold_Schwarzenegger': 19, 'Arnold_Vosloo': 20, 'Ashton_Kutcher': 21, 'Ben_Affleck': 22, 'Ben_Kingsley': 23, 'Ben_McKenzie': 24, 'Ben_Stiller': 25, 'Benicio_Del_Toro': 26, 'Bernard_Hill': 27, 'Bernie_Mac': 28, 'Bill_Cosby': 29, 'Bill_Hader': 30, 'Bill_Murray': 31, 'Bill_Nighy': 32, 'Billy_Bob_Thornton': 33, 'Billy_Boyd': 34, 'Billy_Burke': 35, 'Billy_Zane': 36, 'Brad_Garrett': 37, 'Brad_Pitt': 38, 'Bradley_Cooper': 39, 'Brendan_Fraser': 40, 'Bruce_Greenwood': 41, 'Bruce_Willis': 42, 'Burt_Reynolds': 43, 'Cam_Gigandet': 44, 'Cary_Elwes': 45, 'Casey_Affleck': 46, 'Channing_Tatum': 47, 'Charlie_Day': 48, 'Charlie_Sheen': 49, 'Chazz_Palminteri': 50, 'Chris_Evans': 51, 'Chris_Kattan': 52, 'Chris_Noth': 53, 'Chris_Rock': 54, 'Christian_Bale': 55, 'Christian_Slater': 56, 'Christopher_Lloyd': 57, 'Christopher_Reeve': 58, 'Clint_Eastwood': 59, 'Clive_Owen': 60, 'Colin_Farrell': 61, 'Colin_Firth': 62, 'Colin_Hanks': 63, 'Dan_Lauria': 64, 'Daniel_Craig': 65, 'Daniel_Day-Lewis': 66, 'Daniel_Radcliffe': 67, 'Danny_Glover': 68, 'Danny_Masterson': 69, 'Danny_Trejo': 70, 'David_Boreanaz': 71, 'David_Cross': 72, 'David_Duchovny': 73, 'David_Schwimmer': 74, 'David_Wenham': 75, 'Dean_Cain': 76, 'Denzel_Washington': 77, 'Dermot_Mulroney': 78, 'Desmond_Harrington': 79, 'Diego_Luna': 80, 'Dominic_Monaghan': 81, 'Don_Cheadle': 82, 'Dustin_Hoffman': 83, 'Dwayne_Johnson': 84, 'Ed_Harris': 85, 'Edi_Gathegi': 86, 'Elijah_Wood': 87, 'Emile_Hirsch': 88, 'Eric_Dane': 89, 'Ethan_Hawke': 90, 'Ewan_McGregor': 91, 'Freddy_Prinze_Jr.': 92, 'Freddy_Rodríguez': 93, 'Gabriel_Macht': 94, 'Gary_Dourdan': 95, 'Gary_Oldman': 96, 'Gene_Hackman': 97, 'Geoffrey_Rush': 98, 'George_Clooney': 99, 'George_Lopez': 100, 'Gerard_Butler': 101, 'Giovanni_Ribisi': 102, 'Hal_Holbrook': 103, 'Hank_Azaria': 104, 'Harrison_Ford': 105, 'Harry_Connick_Jr.': 106, 'Hayden_Christensen': 107, 'Heath_Ledger': 108, 'Hector_Elizondo': 109, 'Hugh_Grant': 110, 'Hugh_Jackman': 111, 'Hugo_Weaving': 112, 'Ian_Holm': 113, 'Ian_McKellen': 114, 'Ioan_Gruffudd': 115, 'J.K._Simmons': 116, 'Jack_Nicholson': 117, 'Jackie_Chan': 118, 'Jackson_Rathbone': 119, 'Jaden_Smith': 120, 'Jake_Gyllenhaal': 121, 'Jake_Weber': 122, 'James_Brolin': 123, 'James_Frain': 124, 'James_Franco': 125, 'James_Marsden': 126, 'James_McAvoy': 127, 'James_Remar': 128, 'Jamie_Foxx': 129, 'Jared_Padalecki': 130, 'Jason_Bateman': 131, 'Jason_Behr': 132, 'Jason_Biggs': 133, 'Jason_Lee': 134, 'Jason_Statham': 135, 'Jason_Sudeikis': 136, 'Jay_Baruchel': 137, 'Jean-Claude_Van_Damme': 138, 'Jean_Reno': 139, 'Jeffrey_Tambor': 140, 'Jensen_Ackles': 141, 'Jeremy_Irons': 142, 'Jeremy_Sisto': 143, 'Jerry_Seinfeld': 144, 'Jesse_Eisenberg': 145, 'Jet_Li': 146, 'Jim_Beaver': 147, 'Jim_Carrey': 148, 'Jim_Caviezel': 149, 'Jimmy_Fallon': 150, 'Joaquin_Phoenix': 151, 'Joe_Manganiello': 152, 'Joe_Pantoliano': 153, 'John_Cleese': 154, 'John_Cusack': 155, 'John_Krasinski': 156, 'John_Malkovich': 157, 'John_Noble': 158, 'John_Travolta': 159, 'Johnny_Depp': 160, 'Jon_Hamm': 161, 'Jon_Voight': 162, 'Jonah_Hill': 163, 'Jonathan_Rhys_Meyers': 164, 'Jonathan_Sadowski': 165, 'Josh_Brolin': 166, 'Josh_Duhamel': 167, 'Josh_Hartnett': 168, 'Joshua_Jackson': 169, 'Jude_Law': 170, 'Justin_Long': 171, 'Justin_Timberlake': 172, 'Kal_Penn': 173, 'Karl_Urban': 174, 'Keanu_Reeves': 175, 'Kellan_Lutz': 176, 'Ken_Watanabe': 177, 'Kerr_Smith': 178, 'Kevin_Bacon': 179, 'Kevin_Connolly': 180, 'Kevin_Costner': 181, 'Kevin_McKidd': 182, 'Kiefer_Sutherland': 183, 'Kit_Harington': 184, 'Kris_Kristofferson': 185, 'Laurence_Fishburne': 186, 'Leonardo_DiCaprio': 187, 'Leslie_Neilsen': 188, 'Liev_Schreiber': 189, 'Luke_Wilson': 190, 'Mark_Ruffalo': 191, 'Mark_Wahlberg': 192, 'Martin_Henderson': 193, 'Martin_Lawrence': 194, 'Martin_Sheen': 195, 'Matthew_Broderick': 196, 'Matthew_Gray_Gubler': 197, 'Matthew_Lillard': 198, 'Matthew_McConaughey': 199}
idx2name = {v: k for k, v in name2idx.items()}
BATCH_SIZE = 64
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


class AdvSet(datasets.VisionDataset):
    def __init__(self, adv_dir, real_dir, label_list):
        super(AdvSet, self).__init__(adv_dir)
        self.adv_list = []
        # self.real_dir = real_dir
        self.real_dirs = [os.path.join(real_dir, i) for i in label_list]
        self.real_imgs = []
        self.adv_targets = []

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

        for (root, dirs, files) in os.walk(adv_dir):
            targets = [eval(i.split('-')[0]) for i in files]
            self.adv_targets += targets
            files = [os.path.join(root, i) for i in files]
            self.adv_list += files

        for i in range(len(self.real_dirs)):
            label_dir = self.real_dirs[i]
            files = os.listdir(label_dir)
            files = [os.path.join(label_dir, img) for img in files]
            self.real_imgs.append(files)

    def __getitem__(self, idx):
        img_name = self.adv_list[idx]
        adv = Image.open(img_name)
        adv = self.transform(adv)
        target = self.adv_targets[idx]

        real_list = self.real_imgs[target]
        real = real_list[idx % (len(real_list))]
        real_img = Image.open(real)
        real_img = transforms.ToTensor()(real_img)
        # real_img = self.transform(real_img)
        return adv, real_img, target

    def __len__(self):
        return len(self.adv_list)


def train_inversion():
    adv_dir = './adv_pert'
    real_dir = './data_facescrub'
    # label_name_dict = {'Aaron_Eckhart': 0, 'Adam_Brody': 1, 'Adam_McKay': 2,
    #                    'Adam_Sandler': 3, 'Adrien_Brody': 4, 'Al_Pacino': 5,
    #                    'Alan_Alda': 6, 'Alan_Arkin': 7, 'Alan_Rickman': 8, 'Alec_Baldwin': 9}
    label_name = ['Aaron_Eckhart', 'Adam_Brody', 'Adam_McKay',
                       'Adam_Sandler', 'Adrien_Brody', 'Al_Pacino',
                       'Alan_Alda', 'Alan_Arkin', 'Alan_Rickman', 'Alec_Baldwin', 'Alexander_Skarsgård', 'Alfred_Molina']
    dataset = AdvSet(adv_dir, real_dir, label_name)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = CGAN(3, 3, 6, lambda_L1=10)
    device = torch.device('cuda:0')

    # for epoch in range(100):
    #     for idx, (adv, real_img, target) in enumerate(train_loader):
    #         adv, real_img = adv.to(device), real_img.to(device)
    #         lossD, lossG = model.optimize_parameters(adv, real_img)
    #     print("epoch: ", epoch, "lossD: ", lossD, 'lossG: ', lossG)
    #
    #     if epoch % 10 == 9:
    #         model.save('./model', epoch)
    #         print('model saved')

    res_dir = './result2'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    model.load('./model/GAN_99.pt')
    for idx, (adv, real_img, target) in enumerate(test_loader):
        adv, real_img = adv.to(device), real_img.to(device)
        fake, real = model.forward(adv, real_img)
        fake = fake[0].squeeze().detach().cpu()
        fake_img = transforms.ToPILImage()(fake).convert('RGB')
        fake_img.save(os.path.join(res_dir, '{0}_{1}_fake.png'.format(target, idx)))
        real = real[0].squeeze().detach().cpu()
        real_img = transforms.ToPILImage()(real).convert('RGB')
        real_img.save(os.path.join(res_dir, '{0}_{1}_real.png'.format(target, idx)))



train_inversion()















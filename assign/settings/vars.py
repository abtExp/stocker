import os

from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint

PROJECT_PATH = os.path.abspath(__file__)
PROJECT_PATH = PROJECT_PATH[:PROJECT_PATH.index('assign')].replace('\\', '/')

MODEL_IMAGE_PATH = PROJECT_PATH+'model_images/'

MAX_SENTENCES = 100
MAX_SENTENCE_LENGTH = 200
MAX_SEGMENT_LENGTH = 2.5
MAX_AUDIO_DURATION = 15.0
N_FFT = 1024
HOP_LENGTH = 512
FRAME_RATE = 22050*2
VOCAB_SIZE = 50000
EMBEDDING_SIZE = 768
NUM_SEGMENTS_PER_AUDIO = int(MAX_AUDIO_DURATION//MAX_SEGMENT_LENGTH)
AUDIO_EMBEDDING_SIZE = 3456
NUM_DAYS_PRED = 30
TRAIN_SPLIT = 0.95
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
TRAIN_EPOCHS = 10
STEPS_PER_EPOCH = 100

DATA_PATH = PROJECT_PATH+'data/'
EMBEDDING_FILE = PROJECT_PATH+'data/embeddings/glove.840B.300d.txt'
TOKENIZER_PATH = PROJECT_PATH+'assign/checkpoints/tokenizer.pickle'

YAHOO_DOWNLOAD_FINLINK = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history&crumb={}'
YAHOO_FINLINK = 'https://finance.yahoo.com/quote/{}/history?p={}'

def get_callbacks(model='custom'):
	all_checks = os.listdir(PROJECT_PATH+'assign/checkpoints/')
	all_logs = os.listdir(PROJECT_PATH+'assign/logs/')
	counter = 0
	max = -1

	for folder in all_checks:
			if 'checkpoints_{}'.format(model) in folder:
					if int(folder[folder.rindex('_')+1:]) > max:
							max = int(folder[folder.rindex('_')+1:])

	counter = max+1
	check_path = PROJECT_PATH+'assign/checkpoints/checkpoints_{}_{}/'.format(model, counter)
	logs_path = PROJECT_PATH+'assign/logs/logs_{}_{}/'.format(model, counter)

	if not os.path.isdir(check_path) and not os.path.isdir(logs_path):
			os.mkdir(check_path)
			os.mkdir(logs_path)


	checkpoint = ModelCheckpoint(check_path+'weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=True)
	earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0)
	tensorboard = TensorBoard(log_dir=logs_path, histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)
	reducelr = ReduceLROnPlateau(monitor='loss', factor=0.02, patience=1, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

	return [checkpoint, tensorboard, reducelr, earlystop]


ALL_COMPS = [
	"3M Company",
	"A.O. Smith Corp",
	"ABIOMED Inc",
	"AES Corp",
	"AMETEK Inc.",
	"AT&T Inc.",
	"AbbVie Inc.",
	"Abbott Laboratories",
	"Activision Blizzard",
	"Adobe Systems Inc",
	"Advance Auto Parts",
	"Advanced Micro Devices Inc",
	"Aetna Inc",
	"Agilent Technologies Inc",
	"Akamai Technologies Inc",
	"Alaska Air Group Inc",
	"Alexion Pharmaceuticals",
	"Align Technology",
	"Allegion",
	"Allergan, Plc",
	"Alliance Data Systems",
	"Altria Group Inc",
	"Amazon.com Inc.",
	"Ameren Corp",
	"American Express Co",
	"American Tower Corp A",
	"AmerisourceBergen Corp",
	"Amgen Inc.",
	"Anthem Inc.",
	"Aon plc",
	"Apache Corporation",
	"Archer-Daniels-Midland Co",
	"Autodesk Inc.",
	"Automatic Data Processing",
	"Avery Dennison Corp",
	"Ball Corp",
	"Baxter International Inc.",
	"Becton Dickinson",
	"Biogen Inc.",
	"Boeing Company",
	"Booking Holdings Inc",
	"BorgWarner",
	"Bristol-Myers Squibb",
	"Broadridge Financial Solutions",
	"CA, Inc.",
	"CBRE Group",
	"CBS Corp.",
	"CIGNA Corp.",
	"CME Group Inc.",
	"CMS Energy",
	"CVS Health",
	"Cadence Design Systems",
	"Campbell Soup",
	"Cardinal Health Inc.",
	"Carmax Inc",
	"Caterpillar Inc.",
	"Cboe Global Markets",
	"Celgene Corp.",
	"CenturyLink Inc",
	"Chevron Corp.",
	"Church & Dwight",
	"Cimarex Energy",
	"Cisco Systems",
	"Citigroup Inc.",
	"Citrix Systems",
	"Coca-Cola Company (The)",
	"Cognizant Technology Solutions",
	"Comcast Corp.",
	"Comerica Inc.",
	"Conagra Brands",
	"Corning Inc.",
	"Coty, Inc",
	"Crown Castle International Corp.",
	"Cummins Inc.",
	"D. R. Horton",
	"DTE Energy Co.",
	"DaVita Inc.",
	"Darden Restaurants",
	"Digital Realty Trust Inc",
	"Dish Network",
	"Dollar General",
	"Dominion Energy",
	"Dover Corp.",
	"Duke Energy",
	"EQT Corporation",
	"Ecolab Inc.",
	"Edison Int'l",
	"Edwards Lifesciences",
	"Electronic Arts",
	"Emerson Electric Company",
	"Equinix",
	"Estee Lauder Cos.",
	"Exxon Mobil Corp.",
	"F5 Networks",
	"FLIR Systems",
	"Facebook, Inc.",
	"Fidelity National Information Services",
	"Fifth Third Bancorp",
	"FleetCor Technologies Inc",
	"Foot Locker Inc",
	"Ford Motor",
	"Fortive Corp",
	"Fortune Brands Home & Security",
	"Franklin Resources",
	"Freeport-McMoRan Inc.",
	"Gap Inc.",
	"Garmin Ltd.",
	"Gartner Inc",
	"General Dynamics",
	"General Growth Properties Inc.",
	"General Mills",
	"General Motors",
	"Genuine Parts",
	"Gilead Sciences",
	"Goldman Sachs Group",
	"Goodyear Tire & Rubber",
	"Grainger (W.W.) Inc.",
	"HCA Holdings",
	"HP Inc.",
	"Halliburton Co.",
	"Hanesbrands Inc",
	"Harris Corporation",
	"Hasbro Inc.",
	"Henry Schein",
	"Hess Corporation",
	"Hewlett Packard Enterprise",
	"Hilton Worldwide Holdings Inc",
	"Hologic",
	"Home Depot",
	"Hormel Foods Corp.",
	"Humana Inc.",
	"IPG Photonics Corp.",
	"Illinois Tool Works",
	"Illumina Inc",
	"Incyte",
	"Ingersoll-Rand PLC",
	"Intel Corp.",
	"Intercontinental Exchange",
	"International Paper",
	"Interpublic Group",
	"Intuit Inc.",
	"Invesco Ltd.",
	"Iron Mountain Incorporated",
	"JPMorgan Chase & Co.",
	"Johnson Controls International",
	"Juniper Networks",
	"KLA-Tencor Corp.",
	"Kansas City Southern",
	"Kellogg Co.",
	"Kimberly-Clark",
	"Kinder Morgan",
	"Kohl's Corp.",
	"Kraft Heinz Co",
	"Kroger Co.",
	"LKQ Corporation",
	"Lam Research",
	"Lennar Corp.",
	"Lilly (Eli) & Co.",
	"Lockheed Martin Corp.",
	"Lowe's Cos.",
	"LyondellBasell",
	"MGM Resorts International",
	"MSCI Inc",
	"Marsh & McLennan",
	"Martin Marietta Materials",
	"Masco Corp.",
	"Mastercard Inc.",
	"Mattel Inc.",
	"McDonald's Corp.",
	"Merck & Co.",
	"Michael Kors Holdings",
	"Micron Technology",
	"Microsoft Corp.",
	"Molson Coors Brewing Company",
	"Mondelez International",
	"Motorola Solutions Inc.",
	"Mylan N.V.",
	"NRG Energy",
	"Nasdaq, Inc.",
	"Nektar Therapeutics",
	"NetApp",
	"Newell Brands",
	"Newmont Mining Corporation",
	"News Corp. Class A",
	"NiSource Inc.",
	"Nielsen Holdings",
	"Nike",
	"Noble Energy Inc",
	"Nordstrom",
	"Norfolk Southern Corp.",
	"Norwegian Cruise Line",
	"Nucor Corp.",
	"Nvidia Corporation",
	"Oracle Corp.",
	"PACCAR Inc.",
	"PG&E Corp.",
	"PPG Industries",
	"PPL Corp.",
	"Parker-Hannifin",
	"PayPal",
	"Pentair plc",
	"PepsiCo Inc.",
	"PerkinElmer",
	"Perrigo",
	"Pinnacle West Capital",
	"Polo Ralph Lauren Corp.",
	"Prologis",
	"Public Serv. Enterprise Inc.",
	"Quanta Services Inc.",
	"Quest Diagnostics",
	"Raytheon Co.",
	"Red Hat Inc.",
	"Regeneron",
	"Regions Financial Corp.",
	"Republic Services Inc",
	"ResMed",
	"Rockwell Automation Inc.",
	"Roper Technologies",
	"Ross Stores",
	"Royal Caribbean Cruises Ltd",
	"SCANA Corp",
	"SVB Financial",
	"Salesforce.com",
	"Schlumberger Ltd.",
	"Seagate Technology",
	"Sealed Air",
	"Sempra Energy",
	"Sherwin-Williams",
	"Skyworks Solutions",
	"Snap-On Inc.",
	"Starbucks Corp.",
	"Stericycle Inc",
	"Stryker Corp.",
	"SunTrust Banks",
	"Symantec Corp.",
	"Synopsys Inc.",
	"Sysco Corp.",
	"TE Connectivity Ltd.",
	"Tapestry, Inc.",
	"Target Corp.",
	"Texas Instruments",
	"The Bank of New York Mellon Corp.",
	"The Clorox Company",
	"The Cooper Companies",
	"The Hershey Company",
	"The Mosaic Company",
	"The Walt Disney Company",
	"Thermo Fisher Scientific",
	"Tiffany & Co.",
	"Tractor Supply Company",
	"TransDigm Group",
	"Twitter, Inc.",
	"Tyson Foods",
	"U.S. Bancorp",
	"Ulta Beauty",
	"United Health Group Inc.",
	"United Parcel Service",
	"United Technologies",
	"Valero Energy",
	"Varian Medical Systems",
	"Ventas Inc",
	"Verisign Inc.",
	"Verizon Communications",
	"Viacom Inc.",
	"Vulcan Materials",
	"Walgreens Boots Alliance",
	"Walmart",
	"Waste Management Inc.",
	"Wec Energy Group Inc",
	"Welltower Inc.",
	"WestRock",
	"Western Digital",
	"Western Union Co",
	"Weyerhaeuser",
	"XL Group",
	"Xcel Energy Inc",
	"Xerox",
	"Xilinx",
	"Yum! Brands Inc",
	"eBay Inc."
]
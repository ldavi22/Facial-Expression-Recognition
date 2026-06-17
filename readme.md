# სახის გამომეტყველების ამოცნობა 
 
## პრობლემა
 
დავალება არის მრავალკლასიანი სურათების კლასიფიკაცია **FER2013** მონაცემთა ბაზაზე: 48x48 პიქსელიანი შავთეთრი სახის სურათები, სადაც თითოეული სურათი შეესაბამება **7 ემოციიდან** ერთ-ერთს (გაბრაზებული, ზიზღი, შიში, ბედნიერი, სევდიანი, გაკვირვებული, ნეიტრალური). მონაცემები CSV ფაილის სახითაა წარმოდგენილი, სადაც თითოეული სტრიქონი შეიცავს ემოციას და 2304 პიქსელის მნიშვნელობას სტრიქონის სახით (0-255 შუალედში).
 
დოკუმენტი აღწერს CNN-ის ექსპერიმენტების სრულ პროცესს: ყოველი არქიტექტურული ცვლილება ხდებოდა იტერაციულად და ყოველი run-ის დროს აისახებოდა ეს ყველაფერი. ერთზე მეტი ცვლილება იტერაციებს შორის არ შემქონდა, რათა მივხვედრილიყავი, რა ცვლიდა კონკრეტულად მოდელის შედეგს.
 
## რეპოზიტორის სტრუქტურა
 
```
FER2013-Emotion-Recognition/
├── README.md
├── model_experiment_customCNN.ipynb       # CustomCNN experiments (Runs 1-7, 14%->65%)
├── model_experiment_VGGStyleNet.ipynb     # VGG3StyleNet experiments (67-68%)
├── model_experiment_VGG4StyleNet.ipynb    # VGG4StyleNet experiments (70%)
├── model-experiment-resnettype.ipynb      # ResNet-style with skip connections
└── model_inference.ipynb                  # Inference + Kaggle submission
```
 
## მონაცემთა Pipeline
 
### Dataset კლასი
 
```python
class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        pixels = np.stack([
            np.array(p.split(), dtype=np.float32).reshape(1, 48, 48) / 255.0
            for p in df['pixels'].values
        ])
        self.pixels = torch.tensor(pixels)  # (N, 1, 48, 48)
        self.labels = df['emotion'].values
        self.transform = transform
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        img = self.pixels[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
```
 
**მთავარი გადაწყვეტილებები:**
 
- **პიქსელების წინასწარი გამოთვლა `__init__`-ში, არა `__getitem__`-ში.** თავდაპირველი ვერსია ახდენდა პიქსელების სტრიქონის გარდაქმნას (`.split()` + `np.array(...)`) `__getitem__`-ში, ანუ ეს ოპერაცია ეპოქების განმავლობაში მილიონობით ჯერ მეორდებოდა. `__init__`-ში გადატანა ამ გარდაქმნას მხოლოდ ერთხელ, dataset-ის გამოძახების დროს ასრულებს.
- **ნორმალიზაცია (`/ 255.0`) ყოველთვის გამოიყენება.** ეს იყო მთელი პროექტის ყველაზე მნიშვნელოვანი გამოსწორება (იხ. გაშვება 1) — მის გარეშე მოდელი საერთოდ ვერ სწავლობდა.
### ტრენინგ/ვალიდაციის გაყოფა
 
`train_test_split` პარამეტრებით `test_size=0.2`, `stratify=df['emotion']`, `random_state=42` — Stratify უზრუნველყოფს სიმრავლეებში კლასების თანაბრად გადანაწილებას, ვინაიდან ეს დატასეტი არ არის დაბალანსებული, საჭიროა რომ ტრეინ და ტესტ სეტებში თანაბარი გადანაწილებით იყოს კლასები.
 
---
 
## პირველადი ტესტირების ფუნქცია
 
```python
def sanity_check(model, train_loader, model_name, LR, device, optimizer, criterion,
                  n_classes=7, overfit_batch_size=4):
```
 
ეს ფუნქცია ორ მთავარ რამეს ამოწმებს, იქამდე სანამ მოდელის სრული ტრენინგი დაიწყება. ვინაიდან მოდელის დისფუნქცია ცხადი არ არის და ერორად არ გვევლინება, საჭიროა წინასწარი დიაგნოსტიკა მოდელის.
 
### შემოწმება 1: დანაკარგი პირველი forward run-ის დროს
 
```python
output = model_copy(images)
loss_init = criterion(output, labels)
expected_loss = -torch.log(torch.tensor(1.0 / n_classes)).item()
```
 
ახლად ინიციალიზებული კლასიფიკატორისთვის `n_classes=7` და `CrossEntropyLoss`-ით, loss ინიციალიზაციაზე უნდა იყოს დაახლოებით `-log(1/7) ≈ 1.9459` — ეს არის loss, რომელსაც მიიღებდი თუ მოდელი ყველა კლასზე თანაბარ ალბათობას გასცემდა (ანუ "ჯერ არაფერი იცის, მაგრამ გაფუჭებული არ არის"). თუ რეალური loss ამ მნიშვნელობისგან ძლიერ განსხვავდება, რაღაც არასწორია ინიციალიზაციაში ან არქიტექტურაში — *ტრენინგის დაწყებამდეც*.
 
ასევე ილოგება `prob_sum` (უნდა იყოს 1.0, softmax-ის სწორ მუშაობას ადასტურებს) და `sample_prob` (უნდა იყოს ≈ 1/7 ≈ 0.143 თანაბარი აუთფუთისთვის).
 
### შემოწმება 2: Small Batch Overfitting
 
```python
tiny_images = images[:overfit_batch_size]  
tiny_labels = labels[:overfit_batch_size]
 
for step in range(200):
    output = model_copy(tiny_images)
    loss = criterion(output, tiny_labels)
    optimizer_copy.zero_grad()
    loss.backward()
    optimizer_copy.step()
```
 
თუ მოდელი ვერ ახდენს loss-ის ~0-მდე დაყვანას და სიზუსტის 100%-მდე გაზრდას მხოლოდ **4 ნიმუშზე** 200 იტერაციაში (backward pass), მაშინ გვაქვს კავშირების პრობლემა ( forward pass, backward pass, ან optimizer connection) — და სრულ dataset-ზე ტრენინგი ამას ვერ გამოასწორებს. ეს არის ყველაზე მარტივი ტესტი "მუშაობს თუ არა Gradient Descent".
 
### რატომ მუშაობს deep copy-ზე
 
```python
model_copy = copy.deepcopy(model).to(device)
optimizer_copy = type(optimizer)(model_copy.parameters(), lr=LR)
```
 
მცირე ბაჩზე ოვერფიტის ტესტი **წონებს განაახლებს**. თუ ეს რეალურ `model`/`optimizer` ობიექტებზე გაეშვება, შემდგომი "რეალური" ტრენინგ-რანი დაიწყება მოდელიდან, რომელიც უკვე 4 კონკრეტულ ნიმუშზე დატრენინგებულია და ამასთან ერთად ოპტიმაიზერის პარამეტრებიც შეიცვლება — რეალური run დაბინძურდება. Deep copy-ზე მუშაობა ახალ ოპტიმიზატორთან ერთად sanity check-ს გვერდითი ეფექტების გარეშე ტოვებს.
 
---
 
## Run-by-Run Log
 
### [საწყისი CNN, ნორმალიზაციის გარეშე (underfit)](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/i6kgyc1h?nw=nwuserldavi22)
 
**არქიტექტურა:** 3 conv ფენა (32->64->128 არხი), 2 max-pool, FC(18432->256)->FC(256->7). BatchNorm და Dropout-ის გარეშე.
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning rate | 1e-3 |
| Batch ზომა | 64 |
| Optimizer | Adam |
| Scheduler | არა |
| აუგმენტაცია | არა |
| ეპოქები | 50 |
| შეყვანის ნორმალიზაცია | **არა (0-255 პიქსელები)** |
 
**შედეგი:** train/val სიზუსტე **~14%-ზე** გაჩერებული 50 ეპოქის განმავლობაში. Train loss მერყეობდა 4.319-4.320 ვიწრო ზოლში, არ მცირდებოდა.
 
**დიაგნოზი — მძიმე underfit:** ეს უარესია ვიდრე შემთხვევითი გამოცნობა. მიზეზი: `CustomDataset`-ს `transform=None` ჰქონდა და **ნორმალიზაცია საერთოდ არ გამოიყენებოდა** — 0-255 პიქსელები პირდაპირ სამ conv ფენაზე იკვებებოდა BatchNorm-ის გარეშე. შეგვიძლია ვივარაუდოთ, რომ შემთხვევითმა ინიციალიზაციამ განაპირობა მკვდარი ნეირონების არსებობა, რამაც შემდგომა გამოიწვია მოდელის განუწვრთნელობა და ასეთი შედეგი
 
**გამოსწორება:** პიქსელების [0, 1]-ზე ნორმალიზაცია 255.0-ზე გაყოფით.
 
---
 
### [იგივე მოდელი + ფოტოს ნორმალიზაცია](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/zdgdn5oi?nw=nwuserldavi22)
 
`/ 255.0` ნორმალიზაციის გამოყენების შემდეგ Sanity Check ხელახლა გაეშვა:
 
| სანიტი ჩეკის მეტრიკა | მნიშვნელობა |
|---|---|
| `init_check/loss` | 1.94593 |
| `init_check/expected_loss` | 1.94591 |
| `init_check/prob_sum` | 1.0 |
| `init_check/sample_prob` | 0.1314 |
| `overfit/loss` (200 ნაბიჯის შემდეგ) | 0.0 |
| `overfit/acc` (200 ნაბიჯის შემდეგ) | 1.0 |
 
Loss ინიციალიზაციაზე ახლა თეორიულ `-log(1/7)`-ს თითქმის ზუსტად ემთხვევა, და მოდელს შეუძლია 4 ნიმუშის სრული ოვერფიტი. ეს ადასტურებს, რომ pipeline სწორია — პირველ გაშვებაში მხოლოდ შეყვანის მასშტაბი იყო პრობლემა.
 
ამ რანზე შეიძლება ასევე ითქვას ისიც, რომ გვაქვს ცხადი `overfit`. ``train/acc`` ~ 99%  | `val/acc` ~ 53% , რაც იმაზე მიგვითებს, რომ მოდელს დაზეპირებული აქვს ტრენინგზე შემოსული დატა. სხვაობა ამ ორ მეტრიკას შორის გვიჩვენებს, რომ მოდელი ვარ განაზოგადებს.
 
---
 
### [BatchNorm-ის დამატება](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/a9ecnhm5?nw=nwuserldavi22)
 
 
რატომ `BatchNorm`: ყოველი აქტივაციის წინ ამ ფენის დამატება უზრუნველყოფს ნორმალიზაციას, რის შედეგადაც ქსელი ნაკლებად სენსიტიურია საწყისი ინიციალიზაციების მიმართ ( პრობლემა, რომელიც პირველ გაშვებას ქონდა, ამ ლეიერის გამოყენების შემთხვევაში არ გვექნებოდა ) და ამასთან ერთად აუმჯობესებს კრებადობის (Convergence) სიჩქარეს. 
 
---
 
### [Dropout + weight decay-ის დამატება](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/u7wsgs7u?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch ზომა | 256 |
| Optimizer | Adam |
| Scheduler | არა |
| აუგმენტაცია | არა |
| ეპოქები | 50 |
| არქიტექტურა | 3 conv ბლოკი (32->64->128) BatchNorm + ReLU + Pool, `Dropout2d(0.2)` Conv-ების მერე, `Dropout(0.5)` FC-ს მერე |
 
**შედეგი :** `train/acc` ~ 38%, val/acc ~ 47-48% -- val/acc `train/acc`-ზე მაღალი.
 
**დიაგნოზი — val > train აქ პრობლემა არ არის.** ეს არის მოსალოდნელი ეფექტი Dropout-ისა და BatchNorm-ის სხვადასხვა ქცევისა train vs. eval რეჟიმში:
- ტრენინგის დროს `Dropout(0.5)` FC ერთეულების ~ნახევარს ნულავს, `Dropout2d(0.2)` კი სივრცულ feature map-ებს ნულავს.
- ვალიდაციის დროს dropout გამორთულია და BatchNorm running სტატისტიკას იყენებს -- ორივე eval-რეჟიმის ქსელს *უკეთ* ამუშავებს.
- `total_acc_train` Batch-by-Batch გროვდება *წონების განახლებასთან ერთდროულად*, ამიტომ ადრეული ეპოქები train სიზუსტეს ამახინჯებს.
ეს სხვაობა ჯანსაღია და ჩვეულებრივ მცირდება ტრენინგის პროგრესთან ერთად.
ეს გაშვება მხოლოდ 50 ეპოქაზე ვამუშავე, იმის გათვალისწინებით, რომ მომავალი რანები 80+ ეპოქაზე მაქ გაშვებული, ეს მოდელიც ნორმალურ შედეგს დადებდა უფრო დიდიხანი რომ მემუშავებინა. წესით იქამდე უნდა მემუშავებინა,სანამ ვალიდაციის დანაკარგი ზრდას არ დაიწყებდა.
 
---
 
### [LR Scheduler-ის დამატება](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/p4xv6iis?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning rate (საწყისი) | 1e-3 |
| Weight decay | 1e-4 |
| Batch ზომა | 256 |
| Scheduler | `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` |
| აუგმენტაცია | არა |
| ეპოქები | 80 |
| არქიტექტურა | გაშვება 4-ის იგივე |
 
**რატომ Scheduler:** როდესაც ვალიდაციის დანაკარგი მერყეობს ერთი მნიშვნელობის განმავლობაში `patience` ეპოქის განმავლობაში, Scheduler ამცირებს Learning Rate-ს `factor`-ით, რაც შემდგომ ეხმარება მოდელს განვითარებაში
 
**შედეგი:** Train სიზუსტემ ~65%-ს მიაღწია. Val სიზუსტე 58%-ზე ავიდა.
 
**LR-ის ქცევა:** Scheduler **~6-ჯერ** ამოქმედდა 80 იტერაციის განმავლობაში , LR 1e-3-დან ~1e-5-მდე ჩამოიყვანა.
 
**დასკვნა : `factor=0.5, patience=5` ძალიან აგრესიული იყო.** ვინაიდან 6-ჯერ ამოქმედდა, 64-ჯერ დაიკლო LR-მ, რამაც ხელი შეუშალა მოდელის ტრენინგს, ამიტომ შემდეგი რანების დროს ეს პარამეტრები შევცვალე .
 
---
 
### [მონაცემთა აუგმენტაცია + Scheduler](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/vl2bz2bp?nw=nwuserldavi22) 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning rate (საწყისი) | 1e-3 |
| Weight decay | 1e-4 |
| Batch ზომა | 256 |
| Scheduler | `ReduceLROnPlateau(mode='min', factor=0.7, patience=7, min_lr=1e-5)` |
| აუგმენტაცია | `RandomHorizontalFlip(p=0.5)`, `RandomRotation(10)`, `RandomAffine(translate=(0.05, 0.05))` |
| ეპოქები | 80 |
| არქიტექტურა | 3-ბლოკიანი CNN |
 
**რატომ ეს აუგმენტაცია:**
- **ჰორიზონტალური flip** -- სახეებისთვის უსაფრთხოა (ბილატერალურ-სიმეტრიული), ლეიბელი არ იცვლება.
- **მცირე ბრუნვა (+-10 გრადუსი)** და **მცირე გადაწევა (+-5%)** -- FER2013-ის სახეები სრულყოფილად ცენტრირებული არ არის, ამიტომ მცირე jitter ეხმარება განზოგადებაში.
- გამოტოვებულია: ვერტიკალური flip-ები (სახეებისთვის უაზროა), დიდი ბრუნვები, color jitter (სურათები გრეისქეილია).
**შედეგი 60-80 ეპოქაზე:** `val/loss` კვლავ სტაბილურად მცირდება, `val/acc` **~57-60%-ს** მიაღწია და კვლავ იზრდება. Schedueler **საერთოდ არ ამოქმედდა** -- `val/loss` მუდმივად უმჯობესდებოდა.
 
**დიაგნოზი:** `val/acc` 80-ე ეპოქაზე **პლატოს ვერ მიაღწია** -- მოდელი ჯერ კიდევ არ ოვერფიტავდა, რაც იმას ნიშნავს, რომ მოდელს კიდევ ქონდა შესაძლებლობა, რომ ესწავლა, მაგრამ როგორც უკვე ვახსენე, აქაც არ დავაცადე საკმარისი ხანი მუშაობისთვის, ამის მიზეზი იყო, ის რომ ამ რანების დროს ბევრი ოპერაცია ხდებოდა CPU-ზე და არა GPU-ზე , რის გამოც იწელებოდა რანები, ამიტომ არ ვუშვებდი ბევრ ეპოქაზე, ეს გამოსწორდა შემდგომ რანებში.
 
---
 
### [მე-4 conv ბლოკის დამატება (CustomCNN4) (Run 1)](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/3qfvjnan?nw=nwuserldavi22)
 
[Run 2](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/egmvubbw?nw=nwuserldavi22)
 
**არქიტექტურის ცვლილება:** მე-4 conv ბლოკი (128->256 არხი) BatchNorm + ReLU + Pool-ით, და მეორე FC ფენა:
 
```python
self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
self.bn4 = nn.BatchNorm2d(256)
self.fc1 = nn.Linear(256 * 6 * 6, 256)   # 9216 -> 256
self.fc2 = nn.Linear(256, 256)
self.output = nn.Linear(256, 7)
```
 
**რატომ ეს კონკრეტული ცვლილება:**
- ვინაიდან სამი კონვოლუციის ბლოკი რაღაც მნიშვნელობას ვერ ცდებოდა სიზუსტის მხრივ, გადავწყვიტე რომ ფენების დამატება უკეთესს შედეგს მოგვცემდა,ამის გამო დავამატე მეოთხე კონვოლუციის ბლოკი, რამაღ შემიმცირა FC ფენაზე შემავალი ინფუთების რაოდენობა, მაგრამ მეორე FC-ც დავამატე ზუსტად იმავე მიზნით, მოდელის შედეგის გასაუმჯობესებლად.
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning rate (საწყისი) | 1e-3 |
| Weight decay | 1e-4 |
| ბაჩის ზომა | 256 |
| Scheduler | `ReduceLROnPlateau(mode='min', factor=0.5, patience=5, min_lr=1e-5)` |
| აუგმენტაცია | გაშვება 6-ის იგივე (GPU-ზე) |
| ეპოქები | 150 |
| არქიტექტურა | 4 conv ბლოკი (32->64->128->256), BN+ReLU+Pool+Dropout2d(0.2), FC(9216->256)->Drop->FC(256->256)->Drop->FC(256->7) |
 
**შედეგი 80-ე ეპოქაზე:** `val/acc` - ~62-63%, ``train/acc`` - ~61% -- ორივე კვლავ სტაბილურად უმჯობესდება.
 
**შედეგი 150-ე ეპოქაზე (საბოლოო):** **train სიზუსტე ~70%, ვალიდაციის სიზუსტე ~65%.** Val/acc პლატოს მიაღწია ~100-110-ე ეპოქაზე. Train კი 70%-მდე გააგრძელა, **პირველი მკაფიო, მდგრადი train/val სხვაობა** . **65% val სიზუსტე ამ კონფიგურაციის პრაქტიკული ჭერია.**
 
**დიაგნოზი** -- ადრეული ოვერფიტი , სწორი გაჩერების ადგილი, ამაზე მეტი ტრენინგი უკვე სერიოზულ ოვერფიტში გადაიყვანდა მოდელს. როგორც ვხედავთ 65% ჭერი არის ამ ტიპის არქიტექტურისთვის.
 
---
 
### ინფრასტრუქტურული ცვლილებები
 
- **GPU-ზე აუგმენტაცია.** თავდაპირველად `transform` `CustomDataset.__getitem__`-ში გამოიყენებოდა (CPU-ზე). Kaggle-ის 2xT4-ზე ეს CPU-ს მძიმე დატვირთვის გამომწვევი იყო (CPU 245%, GPU ~29-30%). გამოსწორება: `augment_fn` პარამეტრი `train_model`-ში, `inputs.to(device)` შემდეგ GPU-ზე გამოიყენება.
- **`num_workers=2, pin_memory=True, persistent_workers=True`** ორივე DataLoader-ში.
---
 
## VGGStyleNet ექსპერიმენტები (model_experiment_vggstylenet.ipynb)
 
CustomCNN-ის გზის ამოწურვის შემდეგ (~65% val სიზუსტე), ექსპერიმენტები VGG-სტილის არქიტექტურაზე გადავიდა. VGG-ს ძირითადი იდეა: **ორი 3x3 conv ერთმანეთზე** pooling-ამდე ნაცვლად ერთისა.
 
### არქიტექტურა: VGG3StyleNet (3 ბლოკი, 64->128->256)
 
```python
# თითოეული ბლოკი: conv->BN->ReLU -> conv->BN->ReLU -> pool -> Dropout2d(0.2)
# სივრცული: 48 -> 24 -> 12 -> 6
# FC:        256*6*6=9216 -> 256 -> 7, Dropout(0.5)
```
 
---
 
### [VGG პირველი რანი (baseline)](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/aviyghk8?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| ბაჩის ზომა | 256 |
| ეპოქები | 100 |
 
**შედეგი:** `val/acc` - **60%**, ``train/acc`` - ~74%, train/val სხვაობა ~14%. Val/loss ოსცილირებდა ვიწრო ზოლში ~40-ე ეპოქიდან -- Schedulerს ხშირი ამოქმედება ოპტიმიზაციას არესტაბილიზირებდა.
 
**განხილვა :** 100 ეპოქაზე იყო გაშვებული, მარა ნაადრევად გავთიშე ვინაიდან მე-60 ეპოქაზე უკვე ძალიან დიდი სხვაობა იყო სიზუსტეებს შორის ტრენინგსა და ვალიდაციაზე, ამასთან ერთად გრაფიკიდანაც ჩანს, რომ ~25 ეპოქის მერე ვალიდაციის დანაკარგი ზრდას იწყებს, რაც მიგვანიშნებს იმაზე , რომ მოდელი იწყებს ტრენინგის დაზეპირებას. 
 
---
 
### [VGG მეორე რანი](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/brkvbsq5?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning rate | 1e-3 |
| ეპოქები | 150 |
 
**შედეგი:**  `~val/acc` - **61%**, ``train/acc`` - ~66%, train/val სხვაობა ~5%.
 
**განხილვა** -- იგივე რაც წინა რანში, ესეც ნაადრევად გავთიშე, 30-ე ეპოქის მერე იწყებს `val/loss` ზრდას, საჭიროებს რეგულარიზაციას.  
 
---
 
### [VGGStyleNet + Scheduler + Augmentation ](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/svl9d62q?nw=nwuserldavi22) 
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning rate | 1е-3 |
| Scheduler | `ReduceLROnPlateau(factor=0.5, patience=5)` |
| ეპოქები | 100 |
| აუგმენტაცია | Flip + Rotation + Translate|
 
**შედეგი:** ``train/acc`` ~75% , `val/acc` ~ 67%
 
**დასკვნა** - წინა რანთაც შედარებით უკეთესი შედეგი, აუგმენტაცია კარგად ეხმარება მოდელს განზოგადებაში. 50 ეპოქის მერე `val/loss` იზრდება, მიდის ნელი ოვერფიტი.
 
---
 
### [VGGStyleNet + Scheduler + Augmentation v3 ](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/xxav2lo6?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning rate | 1е-3 |
| Scheduler | `ReduceLROnPlateau(factor=0.5, patience=5)` |
| ეპოქები | 100 |
| აუგმენტაცია | Flip + Rotation + Translate|
 
**შედეგი:** ``train/acc`` ~32% 35-ე ეპოქაზე, `val/acc` ძალიან ხმაურიანია (26-36).
 
**დასკვნა - LR ძალიან მაღალია.** `Train/loss` დაიწყო ~2.6-ზე (რაც საწყისზე მეტია ~1.94 ) — რაც უკვე მიგვანიშნებს, რომ მოდელში პრობლემაა. Optimizer კარგ უბნებს ახტებოდა რაც იწვევდა რხევებს ვალიდაციის `val/loss` `val/acc` მრუდებზე. Adam-ისთვის საწყისი LR 1e-3 და მასზე დაბალი რიცხვები, ამ რანზე გაშვებული LR გაცილებით მაღალია ვიდრე ადამის რეინჯი.
 
---
 
### არქიტექტურა: VGG4StyleNet (4 ბლოკი, 64->128->256->512)
 
VGG3StyleNet-ის ~67-68%-ზე გაჩერების შემდეგ მე-4 ბლოკი დაემატა:
 
```python
# ბლოკი 4: 6x6 -> 3x3, არხები 256->512
# FC: 512*3*3=4608 -> 512 -> 7
```
 
FC შეყვანის ზომა *მცირდება* (4608 vs 9216) conv ტევადობის გაზრდის ფონზე.
 
---
 
### [VGG4StyleNet + Scheduler + Extended Augmentation](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/ws5fcaym?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| ბაჩის ზომა | 256 |
| Scheduler | `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)` |
| აუგმენტაცია | Flip + Rotation + Translate + ColorJitter(brightness=0.2, contrast=0.2) |
| ეპოქები | 100 |
| Dropout2d | 0.2 |
| Dropout FC | 0.5 |
 
**შედეგი:** val/acc **~70%**, ``train/acc`` ~78%, train/val სხვაობა ~8%. Scheduler ორჯერ ამოქმედდა (~95-ე ეპოქაზე) LR 1e-3-დან ~7e-4-მდე ჩამოყავდა, შემდეგ 0.0005-ზე .
 
**რატომ ColorJitter:** ამ ამოცანის დატასეტი Web Scraping არის შედგენილი, რაც იმას ნიშნავს რომ ფოტოები სხვადასხვა განათების პირობებში არის გადაღებული, ამ ტრანსფორმაციის დამატება, მოდელს ეხმარება სხვადასხვა განათების გარჩევაში. `ColorJitter`-ის მხოლოდ ორ პარამეტრს ვიყენებთ (brightness=0.2, contrast=0.2), ვინაიდან ჩვენი ფოტოები შავთეთრია მხოლოდ განათების და კონტრასტის ცვლილება შეგვიძლია.
 
---
 
### [VGG4StyleNet + Stronger Regularization](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/8ai1mxfy?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| learning rate | 1e-3 |
| Weight decay | 5e-4 |
| ბაჩის ზომა | 256 |
| Scheduler | `ReduceLROnPlateau(factor=0.7, patience=7, min_lr=1e-5)` |
| აუგმენტაცია | Flip + Rotation + Translate |
| Dropout2d | 0.3 |
| Dropout FC | 0.5 |
| ეპოქები | 120 |
 
**შედეგი:** `val/acc` **~68%**, `train/val` **~74%** სხვაობა 6%-მდე.
 
**დასკვნა :** Scheduler ამ რანის დროს 5-ჯერ გაეშვა, ტრენინგის განმავლობაში ვალიდაციის დანაკარგი სულ იკლებდა და არ იზრდებოდა, რაც იმას ნიშნავს, რომ მოდელი სულ უმჯობესდებოდა, წინა რანებისგან განსხვავებით რეგულარიზაციამ შეგვიმცირა სხვაობა სიზუსტეებს შორის. ამ შედეგის გამო გამოვიყენე inference-ზე ეს მოდელი.
 
## ResNet-სტილის ექსპერიმენტები (model-experiment-resnettype.ipynb)
 
### მოტივაცია
 
VGG4StyleNet-მა ~70% val სიზუსტეზე მიაღწია პლატოს. შემდეგი არქიტექტურული ნაბიჯი არის **skip კავშირების** (residual connections) დამატება -- ResNet-ში შემოღებული ტექნიკა, რომელიც Vanishing გრადიენტებს ებრძვის.
 
### ძირითადი იდეა: Skip კავშირები
 
ჩვეულებრივ conv ბლოკში, backprop-ის დროს გრადიენტი თანმიმდევრულად გაივლის ყველა conv-სა და აქტივაციის ფუნქციას. ღრმა ქსელებში ეს Vanishing გრადიენტებს იწვევს. Residual ბლოკი shortcut-ს ამატებს:
 
```
output = relu(f(x) + x)
```
 
სადაც `f(x)` ორი conv-ის შემდეგ გავლილი აუთფუთია და `x` არის identity. თუ ბლოკი კონკრეტული შეყვანისთვის სასარგებლო არ არის, `f(x)` ნულთან ახლოს მნიშვნელობების სწავლას შეძლებს და ბლოკი გახდება `relu(0 + x)` -- შეყვანას ხელუხლებლად გაატარებს. გრადიენტს ორი უკუ გზა აქვს: `f(x)`-ის გავლით (რომელმაც შეიძლება შეასუსტოს) და პირდაპირ identity-ის გავლით (რომელიც უცვლელად გადასცემს).
 
### 1x1 Conv პროექცია
 
როცა ბლოკებს შორის არხების რაოდენობა იცვლება (მაგ. 64->128), `x`-ს და `f(x)`-ს განსხვავებული ფორმა აქვს და პირდაპირ ვერ ემატება. shortcut გზაზე **1x1 conv** წყვეტს ამ პრობლემას -- `x`-ს საჭირო არხების რაოდენობამდე უკეთებს პროექციას, სივრცული ცვლილების გარეშე.
 
```python
self.shortcut = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 1),
    nn.BatchNorm2d(out_channels)
) if in_channels != out_channels else nn.Identity()
```
 
როცა არხები არ იცვლება, `nn.Identity()` გამოიყენება -- ნულოვანი პარამეტრები, `x` უცვლელად გადაეცემა.
 
### არქიტექტურა
 
```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_a   = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn_a     = nn.BatchNorm2d(out_channels)
        self.conv_b   = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn_b     = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
 
    def forward(self, x):
        identity = self.shortcut(x)               # conv გზამდე x-ის შენახვა
        x = self.relu(self.bn_a(self.conv_a(x)))
        x = self.bn_b(self.conv_b(x))             # დამატებამდე ReLU არ არის
        x = self.relu(x + identity)               # დამატება, შემდეგ ReLU
        return x
```
 
Pooling ხდება ResBlock-ს **გარეთ** `Net.forward`-ში, დამატების შემდეგ.
 
4-ბლოკიანი სტრუქტურა VGG4StyleNet:
 
```
ბლოკი 1: ResBlock(1->64)   + pool  -> 24x24
ბლოკი 2: ResBlock(64->128) + pool  -> 12x12
ბლოკი 3: ResBlock(128->256)+ pool  -> 6x6
ბლოკი 4: ResBlock(256->512)+ pool  -> 3x3
FC:      4608 -> 512 -> 7
```
 
### [ResNet-style baseline](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/xcnqb50r?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning Rate | 1e-3 |
| Scheduler | არა |
| აუგმენტაცია | არა |
| ეპოქები | 80 |
 
**შედეგი:** `train/acc` ~94%, `val/acc` ~63%, `train/val` სხვაობა ~31%.
 
**დიაგნოზი — მძიმე ოვერფიტი.** ResNet-ის არქიტექტურა VGG4-ზე გაცილებით მეტ სიმძლავრეს ამჟღავნებს და რეგულარიზაციის გარეშე ტრენინგ სეტს თითქმის სრულად იზეპირებს. 31%-იანი სხვაობა ყველა დიდია, რაც ყოფილა ჯერჯერობით. საჭიროა აუგმენტაცია და/ან Scheduler.
 
---
 
### [ResNet-style + Scheduler](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/el5u70mj?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning Rate | 1e-3 |
| Scheduler | ReduceLROnPlateau f=0.7 p=7|
| აუგმენტაცია | არა |
| ეპოქები | 100 |
 
**შედეგი:** `train/acc` ~91%, `val/acc` ~63%, `train/val` სხვაობა ~28%.
 
**დიაგნოზი :** მხოლოდ 37 ეპოქა იმუშავა და მალევე წავიდა ოვერფიტში, `val/loss` დაახლოებით მე-20 ეპოქის მერე იწყებს ზრდას. Scheduler ამ შემთხვევაში დიდად ვერ გვეხმარება.
 
---
 
### [ResNet-style + Scheduler + Augmentation](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/3vihzclm?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning Rate | 1e-3 |
| Scheduler | `ReduceLROnPlateau(factor=0.7, patience=7, min_lr=1e-5)` |
| აუგმენტაცია | Flip + Rotation + Translate |
| ეპოქები | 120 |
 
**შედეგი:** `train/acc` ~80%, `val/acc` ~69%, `train/val` სხვაობა ~11%.
 
**დიაგნოზი.** აუგმენტაციამ დრამატული ცვლილება გამოიწვია — `val/acc` 63%-დან 69%-მდე გაიზარდა და სხვაობა ~28%-დან ~11%-მდე შემცირდა. სხვაობა კვლავ შესამჩნევია, საჭიროებს მეტ რეგულარიზაციას.
 
---
 
### [ResNet-style + Scheduler + Extended Augmentation](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/ow9a5kuz?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning Rate | 1e-3 |
| Scheduler | `ReduceLROnPlateau(factor=0.7, patience=7, min_lr=1e-5)` |
| აუგმენტაცია | Flip + Rotation + Translate + ColorJitter + RandomErasing(p=0.3) |
| ეპოქები | 120 |
 
**შედეგი:** `train/acc `~83%, `val/acc` ~70%, `train/val` სხვაობა ~13%.
 
**დიაგნოზი.** `val/acc` VGG4StyleNet-ის საუკეთეს შედეგს დაემთხვა (70%). სხვაობა R3-თან შედარებით ოდნავ გაიზარდა (13% vs 11%) — RandomErasing-მა ტრენინგი გაართულა, მაგრამ განზოგადება პროპორციულად ვერ გააუმჯობესა.
 
---
 
### [ResNet-style + Scheduler + Extended Augmentation + Heavy Regularization](https://wandb.ai/ldavi22-free-university-of-tbilisi-/Emotion%20Recognition/runs/213zk974?nw=nwuserldavi22)
 
| პარამეტრი | მნიშვნელობა |
|---|---|
| Learning Rate | 1e-3 |
| Scheduler | `ReduceLROnPlateau(factor=0.7, patience=7, min_lr=1e-5)` |
| აუგმენტაცია | Flip + Rotation + Translate + ColorJitter + RandomErasing |
| Dropout2d | 0.3 |
| Dropout FC | 0.5 |
| ეპოქები | 120 |
 
**შედეგი:** train/acc ~76%, val/acc ~69%, train/val სხვაობა ~7%.
 
**დიაგნოზი.** dropout2d-ის 0.3-მდე გაზრდამ სხვაობა ~13%-დან ~7%-მდე შეამცირა — ResNet ექსპერიმენტებიდან ყველაზე სტაბილური რანი. წინა რანის `val/acc` -ს 1%-ით ჩამოუვარდება (69% vs 70%), მაგრამ train/val სხვაობა გაცილებით უკეთესია.
 
---
 
## შემაჯამებელი ცხრილი — ყველა გაშვება
 
| გაშვება | მოდელი | LR | Scheduler. | აუგ. | ეპ. | Train | Val | შენიშვნა |
|---|---|---|---|---|---|---|---|---|
| 1 | 3-conv, **ნორმ. გარეშე** | 1e-3 | -- | -- | 50 | ~14% | ~14% | **არ მუშაობს** |
| 2 | 3-conv, ნორმ. | 1e-3 | -- | -- | 50 | up | up |  |
| 3 | + BN v1 (BN->Pool->ReLU) | 1e-4 | -- | -- | 50 | -- | -- | BN განლაგება |
| 3b | + BN v2 (BN->ReLU->Pool) | 2e-3 | -- | -- | 50 | -- | -- | BN განლაგება |
| 4 | 3-conv + BN + Dropout | 1e-3 | -- | -- | 50 | ~38% | ~47-50% | Val > Train (Dropout/BN ეფექტი) |
| 5 | + RLP(0.5/5) | 1e-3 | RLP | -- | 80 | ~65% | ~50% | Scheduler. ძალიან აგრ., LR ჩამოვარდა |
| 6 | + RLP.(0.7/7) + აუგ | 1e-3 | RLP | Flip+Rot+Trans | 80 | ~50% | ~57-60% | ოვერფიტი არ არის |
| 7 | **CustomCNN4** (32-64-128-256) | 1e-3 | RLP | GPU | 150 | ~70% | **~65%** | CustomCNN-ის ზედა ზღვარიო\ |
| V1 | VGG3StyleNet baseline | 1e-3 | RLP(0.5/5) | -- | 100 | ~74% | ~60% | სხვაობა ~14%, ნაადრევი ოვერფიტი |
| V2 | VGG3StyleNet | 1e-3 | RLP(0.7/7) | -- | 150 | ~66% | ~61% | val/loss ~30 ეპოქაზე იზრდება |
| V3 | VGG3StyleNet + Scheduler + აუგ | 1e-3 | RLP(0.5/5) | Flip+Rot+Trans | 100 | ~75% | ~67% | VGG3-ის საუკეთესო |
| V4 | VGG3StyleNet, LR=3e-3 | 3e-3 | RLP | Flip+Rot+Trans | ~35 | ~32% | ხმაურიანი | LR ძალიან მაღალი |
| V5 | **VGG4StyleNet** + ColorJitter | 1e-3 | RLP(0.5/5) | +ColorJitter | 100 | ~78% | **~70%** | კარგი შედეგი |
| V6 | VGG4StyleNet + გამ. რეგ. | 1e-3 | RLP(0.7/7) | Flip+Rot+Trans | 120 | ~74% | ~69% | სხვაობა 6%, submission |
| R1 | ResNet-style baseline | 1e-3 | -- | -- | 80 | ~94% | ~63% | მძიმე ოვერფიტი,  სხვაობა 31% |
| R2 | ResNet-style + Scheduler. | 1e-3 | RLP(0.7/7) | -- | 100 | ~91% | ~63% | კვლავ ოვერფიტი |
| R3 | ResNet-style + აუგ | 1e-3 | RLP(0.7/7) | Flip+Rot+Trans | 120 | ~80% | ~69% | საჭიროებს მეტ რეგულარიზაციას|
| R4 | ResNet-style + გაძლიერებული აუგმენტაცია | 1e-3 | RLP(0.7/7) | +Jitter+Erasing | 120 | ~83% | ~70% | კარგი შედეგი |
| R5 | ResNet-style + გაძლიერებული რეგულარიზაცია | 1e-3 | RLP(0.7/7) | +Jitter+Erasing | 120 | ~76% | ~69% | სტაბილური, სხვაობა 7% |
 
RLP = ReduceLROnPlateau
 
---
 
## ძირითადი დასკვნები
 
1. **შეყვანის ნორმალიზაცია სავალდებულოა.** 0-255 პიქსელები BatchNorm-ის გარეშე ReLU-ებს "კლავს" -- ერთი ხაზი (`/255.0`) მოდელი 14%-დან სრული ტრენინგის შესაძლებლობამდე გაიყვანა.
2. **Sanity Check(`loss@init` + 4 ნიმუშზე ოვერფიტი) სანდოდ განასხვავებს "Pipeline bug"-s  "Needs more training"-სგან.**
3. **Val სიზუსტე train-ზე მაღალი ადრეულ ეტაპზე ნორმალურია**, Dropout/BatchNorm-ის ტრენინგ vs. eval რეჟიმში განსხვავებული ქცევის გამო.
4. **LR Schedulerს აგრესიულობა დიდ მნიშვნელობას ატარებს.** `factor=0.5, patience=5` LR-ს 64-ჯერ ამცირებს ~40 ეპოქაში. `factor=0.7, patience=7, min_lr=1e-5` უფრო რბილია.
6. **VGG-სტილის double-conv ბლოკები single-conv-ებს მუდმივად ჯობნის.**
7. **ყოველ გაშვებამდე model, optimizer და scheduler-ის ხელახლა ინიციალიზება.** - წინააღმდეგ შემთხვევაში წინა რანის პარამეტრები გადმოგვაქ
9. **ResNet-სტილის არქიტექტურა VGG4-ზე მეტ რეგულარიზაციას საჭიროებს.** baseline-ზე 31%-იანი train/val სხვაობა მიუთითებს, რომ skip კავშირები მოდელის შესაძლებლობას ( Capacity ) მნიშვნელოვნად ზრდის.


### [WandB Report](https://api.wandb.ai/links/ldavi22-free-university-of-tbilisi-/87w81jgz)
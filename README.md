# flutter_receipt

A Flutter app that runs YOLO detection using a TF-lite formatted model.

## How to test it?

1.Build and Run the app on a real device.<br><br>
2.Open the following picture and take a SS using the app to start a YOLO detection process.<br><br>
flutter_receipt/yolo11n_saved_model/assets/test.jpg<br>
3.You’ll see a cropped portion of the image captured by the YOLO model if the detection is successful.<br><br>



## Model and training data

The bundled model (`models/cord_yolo11n.tflite`) is a YOLO11n detector trained with
[Ultralytics](https://github.com/ultralytics/ultralytics) on the
**CORD-v2** receipt dataset
([naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)).

- **Dataset:** CORD – A Consolidated Receipt Dataset for Post-OCR Parsing, by NAVER CLOVA.
  800 train / 100 validation / 100 test receipt photos from Indonesian shops and restaurants.
- **License:** CORD is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  The dataset itself is not included in this repository; it is downloaded at training time.
- **Conversion:** [training/cord_to_yolo.py](training/cord_to_yolo.py) turns CORD's word-level
  annotations into one YOLO box per line, covering only the value words (e.g. the amount, not the
  word "TOTAL"). 17 of CORD's categories are kept; rare or vague ones are dropped.
- **Training:** [training/cord_yolo_colab.ipynb](training/cord_yolo_colab.ipynb) runs the
  conversion, training and TFLite export on Google Colab (free T4 GPU, about 1 hour).

### Classes

`menu.nm`, `menu.cnt`, `menu.price`, `menu.unitprice`, `menu.discountprice`, `menu.num`,
`menu.sub.nm`, `menu.sub.cnt`, `menu.sub.price`, `sub_total.subtotal_price`,
`sub_total.tax_price`, `sub_total.service_price`, `total.total_price`, `total.cashprice`,
`total.changeprice`, `total.creditcardprice`, `total.menuqty_cnt`

### Results (CORD test split, 100 images, imgsz 640)

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.84 | 0.75 | 0.82 | 0.63 |

### Limitations

- Trained only on Indonesian receipts (prices like `16,500`). Accuracy drops noticeably on other
  formats such as US receipts (`$24.00`): totals can be missed and item prices mislabelled.
- No date, tip or store-name classes: CORD blurs dates and store details for privacy.

### Citation

```bibtex
@inproceedings{park2019cord,
  title     = {CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author    = {Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk},
  booktitle = {Document Intelligence Workshop at Neural Information Processing Systems},
  year      = {2019}
}
```

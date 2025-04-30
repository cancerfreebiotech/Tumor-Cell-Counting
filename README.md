# Tumor Cell Counting

## Environment

Tested on `python 3.7.3`

## Dependency

```
skimage
matplotlib
numpy
pandas
cv2
PIL
tqdm
```

## Usage

Under folder  `./src` :

```
python main.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH]
```

* `-h`: help message. 
* `-i`: root folder of the input images. 
* `-o`: output directory, default is `./output`
## Example

```
python main.py -i '..\demo\(G)PanCK(R)Vimentin(B)Hoechst\S-200420-02\'
```

```
python main.py -i '..\demo\(G)PanCK(R)Vimentin(B)Hoechst\'
```
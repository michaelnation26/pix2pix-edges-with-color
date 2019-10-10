const IMG_WIDTH = 256;
const IMG_HEIGHT = 256;
const IMG_BASE_URL = "source_images/";
const IMG_FILENAMES = [
  "01.jpg",
  "109572.9.jpg",
  "7144144.151.jpg",
  "7423232.2125.jpg",
  "7498391.406782.jpg",
  "7525601.367468.jpg",
];

var sketchpadSource;
var sketchpadTarget;
var model;
var img_idx = -1;

(async () => {
  model = await tf.loadLayersModel('http://0.0.0.0:8001/gen_model/model.json');
  $("#generate").prop("disabled", false);
  $("#generate").html("generate");
})();

$(document).ready(function() {
  sketchpadSource = new Sketchpad({
    element: "#sketchpad-source",
    width: IMG_WIDTH,
    height: IMG_HEIGHT,
    penSize: 1
  });

  sketchpadTarget = new Sketchpad({
    element: "#sketchpad-target",
    width: IMG_WIDTH,
    height: IMG_HEIGHT,
    readOnly: true
  });

  const imgUrl = getImgSourceUrl();
  loadSourceImg(imgUrl);
});

function brush(event) {
  sketchpadSource.penSize = 2;
  sketchpadSource.color = $("#color-picker").val();
  $("#color-picker").removeClass("d-none");
}

function clearSketchPad(sketchpad) {
  sketchpad.clear();
  sketchpadSource.strokes = [];
  sketchpadSource.undoHistory = [];
  sketchpadSource.image = null;
}

function clearSketchPads(event) {
  clearSketchPad(sketchpadSource);
  clearSketchPad(sketchpadTarget);
}

function eraser(event) {
  sketchpadSource.penSize = 8;
  sketchpadSource.color = "#ffffff";
  $("#color-picker").addClass("d-none");
}

function generate(event) {
  const imgSource = getImgSource();
  const imgTarget = generateImgTarget(imgSource);

  const imageData = sketchpadSource.context.createImageData(
    sketchpadSource._width, sketchpadSource._height);
  updateImageDataWithImg3D(imageData, imgTarget);
  sketchpadTarget.context.putImageData(imageData, 0, 0);
}

function generateImgTarget(imgSource) {
  const imgsSource = tf.expandDims(imgSource); // batch of 1
  const imgsTarget = model.predict(imgsSource);
  let imgTarget = tf.squeeze(imgsTarget);
  imgTarget = tf.div(tf.add(imgTarget, tf.scalar(1.0)), tf.scalar(2.0));
  imgTarget = tf.mul(imgTarget, tf.scalar(255.0)).toInt();

  return imgTarget.dataSync();
}

function getImgSource() {
  const imgSourceData = sketchpadSource.context.getImageData(
    0, 0, sketchpadSource._width, sketchpadSource._height);

  return preprocessImg(imgSourceData);
}

function getImgSourceUrl() {
  // increment img index
  img_idx = (img_idx + 1) % IMG_FILENAMES.length;

  return IMG_BASE_URL + IMG_FILENAMES[img_idx];
}

function loadSourceImg(url) {
  const image = new Image();
  image.onload = function () {
      sketchpadSource.image = this;
      sketchpadSource.redraw();
  };
  image.src = url;
}

function newImg() {
  clearSketchPads();

  const imgUrl = getImgSourceUrl();
  loadSourceImg(imgUrl);
}

function pencil(event) {
  sketchpadSource.penSize = 1;
  sketchpadSource.color = "#000000";
  $("#color-picker").addClass("d-none");
}

function preprocessImg(imgData) {
  const input = tf.browser.fromPixels(imgData);
  const inputData = input.dataSync();
  const floatInput = tf.tensor3d(inputData, input.shape, 'float32');
  const normalizedInput = tf.div(floatInput, tf.scalar(255.0));
  const preprocessedInput = tf.sub(tf.mul(normalizedInput, tf.scalar(2)), tf.scalar(1));

  return preprocessedInput;
}

function redo() {
  sketchpadSource.redo();
}

function size(event) {
  sketchpadSource.penSize = $(event.target).val();
}

function undo() {
  sketchpadSource.undo();
}

// Flat 3D Image (RGB) -> ImageData (RGBA)
function updateImageDataWithImg3D(imageData, img3D) {
  const data = imageData.data;
  for (let img3DIdx = 0, imgDataIdx = 0; img3DIdx < img3D.length; img3DIdx += 3, imgDataIdx += 4) {
    data[imgDataIdx] = img3D[img3DIdx];
    data[imgDataIdx+1] = img3D[img3DIdx+1];
    data[imgDataIdx+2] = img3D[img3DIdx+2];
    data[imgDataIdx+3] = 255;
  }
}

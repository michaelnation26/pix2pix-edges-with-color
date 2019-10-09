const IMAGE_WIDTH = 256;
const IMAGE_HEIGHT = 256;
var sketchpadSource;
var sketchpadTarget;
var model;

(async () => {
  model = await tf.loadLayersModel('http://0.0.0.0:8001/models/gen_model/model.json');
  $("#generate").prop("disabled", false);
  $("#generate").html("generate");
})();

$(document).ready(function() {
  sketchpadSource = new Sketchpad({
    element: "#sketchpad-source",
    width: IMAGE_WIDTH,
    height: IMAGE_HEIGHT,
    penSize: 1,
    imageUrl: "source_images/01.jpg"
  });

  sketchpadTarget = new Sketchpad({
    element: "#sketchpad-target",
    width: IMAGE_WIDTH,
    height: IMAGE_HEIGHT,
    readOnly: true
  });

  loadSourceImage("source_images/01.jpg")
});

function clearSketchPad(sketchpad) {
  sketchpad.clear();
  sketchpad.strokes = [];
  sketchpad.undoHistory = [];
  sketchpad.image = null;
}

function clearSketchPads(event) {
  clearSketchPad(sketchpadSource);
  clearSketchPad(sketchpadTarget);
}

function brush(event) {
  sketchpadSource.penSize = 2;
  sketchpadSource.color = $("#color-picker").val();
  $("#color-picker").removeClass("d-none");
}

function eraser(event) {
  sketchpadSource.penSize = 8;
  sketchpadSource.color = "#ffffff";
  $("#color-picker").addClass("d-none");
}

function generate(event) {
  let imgSourceData = sketchpadSource.context.getImageData(
    0, 0, sketchpadSource._width, sketchpadSource._height);

  let imgSource = preprocessImg(imgSourceData);
  let imgsSource = tf.expandDims(imgSource); // batch of 1
  let imgsTarget = model.predict(imgsSource);
  let imgTarget = tf.squeeze(imgsTarget);
  imgTarget = tf.div(tf.add(imgTarget, tf.scalar(1)), tf.scalar(2));
  imgTarget = tf.mul(imgTarget, tf.scalar(255)).toInt();
  imgTarget = imgTarget.dataSync();

  var imageData = sketchpadSource.context.createImageData(
    sketchpadSource._width, sketchpadSource._height);
  updateImageDataWithImg3D(imageData, imgTarget);
  sketchpadTarget.context.putImageData(imageData, 0, 0);
}

// Flat 3D Image (RGB) -> ImageData (RGBA)
function updateImageDataWithImg3D(imageData, img3D) {
  let data = imageData.data;
  for (let img3DIdx = 0, imgDataIdx = 0; img3DIdx < img3D.length; img3DIdx += 3, imgDataIdx += 4) {
    data[imgDataIdx] = img3D[img3DIdx];
    data[imgDataIdx+1] = img3D[img3DIdx+1];
    data[imgDataIdx+2] = img3D[img3DIdx+2];
    data[imgDataIdx+3] = 255;
  }
}

function loadSourceImage(url) {
  let image = new Image();
  image.onload = function () {
      sketchpadSource.image = this;
      sketchpadSource.redraw();
  };
  image.src = url;
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

function preprocessImg2(imgData) {
  let data = imgData.data;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = (data[i] - 127.5) / 127.5;
    data[i+1] = (data[i+1] - 127.5) / 127.5;
    data[i+2] = (data[i+2] - 127.5) / 127.5;
  }

  return imgData;
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

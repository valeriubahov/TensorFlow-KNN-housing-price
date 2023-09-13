import("@tensorflow/tfjs-node");

const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  // get the values to standardize our data
  const { mean, variance } = tf.moments(features, 0);

  // standardize our prediction point
  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  // standardize all our features
  const scaledFeatures = features.sub(mean).div(variance.pow(0.5));

  return (
    // sum += (features - predictionPoint) ** 2
    // prediction = Square root of sum
    scaledFeatures
      .sub(scaledPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1) // transform the tensor from 1D to 2D
      .concat(labels, 1)
      .unstack() // transform tensors into play javascript objects (arrays)
      .sort((a, b) => {
        return a.get(0) > b.get(0) ? 1 : -1;
      })
      .slice(0, k)
      .reduce((acc, pair) => {
        return acc + pair.get(1);
      }, 0) / k
  );
}

let {
  features, // lat and long of our training data
  labels, // price for our data
  testFeatures, // lat and long for testing set
  testLabels, // labels for testing set
} = loadCSV("kc_house_data.csv", {
  shuffle: true, // shuffle the data in order to retrive training and testing data randomly
  splitTest: 10, // return the number of records used for test set, the other records remaining are the training set
  dataColumns: ["lat", "long", "sqft_lot", "sqft_living"], // what features to use for training/test data
  labelColumns: ["price"], // what is our label and values to add in the labels set
});

// generate tensors from our data
features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, index) => {
  const result = knn(features, labels, tf.tensor(testPoint), 20);

  const err = (testLabels[index][0] - result) / testLabels[0][0];

  console.log(`The predicted value of the house is $${result}`);
  console.log(`The real value in that area is $${testLabels[index][0]}`);
  console.log(`The prediction is ${Math.round(100 - Math.abs(err * 100))}% accurate`, "\n");
});

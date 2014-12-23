package ch.eonum.pipeline.classification.lstm;

import ch.eonum.pipeline.core.Sequence;

/**
 * Weighted LSTM which can be used for boosting.
 * All weights are normalized to sum up to the size of the training data.
 * 
 * @author tim
 *
 */
public class WeightedLSTM<E extends Sequence> extends LSTM<E> {
	@Override
	protected LSTMCore<E> createNet(String name, String folder, int seed) {
		double[] weights = new double[this.trainingDataSet.size()];
		double[] weightsTest = new double[this.testDataSet.size()];
		int i = 0;
		double sum = 0.0;
		for(Sequence each : this.trainingDataSet){
			weights[i] = each.weight;
			sum += weights[i];
			i++;
		}
		i = 0;
		for(Sequence each : this.testDataSet)
			weightsTest[i++] = each.weight;
		double norm = sum/trainingDataSet.size();
		for(i = 0; i < weights.length; i++)
			weights[i] /= norm;
		for(i = 0; i < weightsTest.length; i++)
			weightsTest[i] /= norm;
		
		return new WeightedLSTMCore<E>(name, folder, this, seed, outputGates, forgetGates, inputGates, weights, weightsTest);
	}
}

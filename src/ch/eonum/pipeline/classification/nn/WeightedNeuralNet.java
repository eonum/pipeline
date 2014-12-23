package ch.eonum.pipeline.classification.nn;

import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;

/**
 * Allows the training with weighted instances and hence the use of meta
 * classifiers like Boosting.
 * 
 * @author tim
 * 
 * @param <E>
 */
public class WeightedNeuralNet<E extends Instance> extends NeuralNet<E> {

	public WeightedNeuralNet(Features features) {
		super(features);
	}
	
	@Override
	protected NeuralNetCore<E> createNet(String name, int seed) {
		double[] weights = new double[this.trainingDataSet.size()];
		double[] weightsTest = new double[this.testDataSet.size()];
		int j = 0;
		double sum = 0.0;
		for(Instance each : this.trainingDataSet){
			weights[j] = each.weight;
			sum += weights[j];
			j++;
		}
		j = 0;
		for(Instance each : this.testDataSet)
			weightsTest[j++] = each.weight;
		double norm = sum/trainingDataSet.size();
		for(j = 0; j < weights.length; j++)
			weights[j] /= norm;
		for(j = 0; j < weightsTest.length; j++)
			weightsTest[j] /= norm;
		return new WeightedNeuralNetCore<E>(name, this.getBaseDir() + name
				+ "/", this, seed * 11, weights, weightsTest, dropout, classify);
	}

}

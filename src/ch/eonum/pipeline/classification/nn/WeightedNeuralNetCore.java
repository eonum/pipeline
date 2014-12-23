package ch.eonum.pipeline.classification.nn;

import ch.eonum.pipeline.core.Instance;

/**
 * Same functionality as @see NeuralNetCore but takes the weights of training
 * instances into account.
 * 
 * @author tim
 * 
 * @param <E>
 */
public class WeightedNeuralNetCore<E extends Instance> extends NeuralNetCore<E> {

	private double[] weights;
	private double[] weightsTest;

	public WeightedNeuralNetCore(String threadName, String baseDir,
			NeuralNet<E> parent, int seed, double[] weights, double[] weightsTest, boolean dropout, boolean classify) {
		super(threadName, baseDir, parent, seed, dropout, classify);
		this.weights = weights;
		this.weightsTest = weightsTest;
	}
	
	@Override
	protected void updateWeights() {
		for (int i = 0; i < deltaInputHidden.length; i++)
			for (int j = 0; j < deltaInputHidden[i].length; j++)
				deltaInputHidden[i][j] *= weights[example];
		for (int i = 0; i < deltaHiddenOutput.length; i++)
			for (int j = 0; j < deltaHiddenOutput[i].length; j++)
				deltaHiddenOutput[i][j] *= weights[example];

		super.updateWeights();
	}
	
	@Override
	protected double getSetAccuracy(double[][] inputs, double[][] targets) {
		double mse = 0;
		double weightSum = 0.;
		for (int tp = 0; tp < (int) inputs.length; tp++) {
			feedForward(inputs[tp]);
			weightSum += weightsTest[tp];
			for (int k = 0; k < nOutput; k++)
				mse += Math.pow((outputNeurons[k] - targets[tp][k]), 2) * weightsTest[tp];
		}
		return mse / (nOutput * weightSum);
	}

}

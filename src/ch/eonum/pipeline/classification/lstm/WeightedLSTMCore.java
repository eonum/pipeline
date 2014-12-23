package ch.eonum.pipeline.classification.lstm;

import ch.eonum.pipeline.core.Sequence;

public class WeightedLSTMCore<E extends Sequence> extends LSTMCore<E> {

	private double[] weights;
	private double[] weightsTest;

	public WeightedLSTMCore(String threadName, String baseDir, LSTM<E> parent,
			int seed, boolean outputGates, boolean forgetGates,
			boolean inputGates, double[] weights, double[] weightsTest) {
		super(threadName, baseDir, parent, seed, outputGates, forgetGates, inputGates, false);
		this.weights = weights;
		this.weightsTest = weightsTest;
	}
	
	@Override
	protected void adjustAlpha(int example) {
		alpha *= this.weights[example];
	}
	
	@Override
	protected double compError(int currentSequence, boolean test) {
		return super.compError(currentSequence, test)
				* (test ? weightsTest[currentSequence]
						: weights[currentSequence]);
	}

}

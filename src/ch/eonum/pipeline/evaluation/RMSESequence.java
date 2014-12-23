package ch.eonum.pipeline.evaluation;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Sequence;

/**
 * Root mean square error for sequences.
 * 
 * @author tim
 *
 */
public class RMSESequence<E extends Sequence> implements Evaluator<E> {

	public RMSESequence(){}

	@Override
	public double evaluate(DataSet<E> dataset) {
		return this.calculateEpsilon(dataset);
	}

	private double calculateEpsilon(DataSet<E> dataset) {
		double epsilon = 0.0;
		double n = 0;
		for(Sequence seq : dataset){
			int outsize = seq.outputSize();
			for(int x = 0; x < seq.getSequenceLength(); x++)
				for(int y = 0; y < outsize; y++){
					if(Double.isNaN(seq.groundTruthAt(x, y)))
						continue;
					n++;
					epsilon += Math.pow(seq.resultAt(x, y) - seq.groundTruthAt(x, y), 2.0);
				}
		}
		epsilon /= n;
		epsilon = Math.sqrt(epsilon);
		return -epsilon;
	}

	@Override
	public void printResults(String fileName) {}

	@Override
	public void printResultsAndGnuplot(String fileName) {}

}

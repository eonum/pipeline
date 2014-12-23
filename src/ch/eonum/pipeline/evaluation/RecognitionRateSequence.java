package ch.eonum.pipeline.evaluation;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Sequence;

public class RecognitionRateSequence<E extends Sequence> implements Evaluator<E> {

	@Override
	public double evaluate(DataSet<E> dataset) {
		double ok = 0.0;
		double n = 0;
		for(Sequence seq : dataset){
			for(int x = 0; x < seq.getGroundTruthLength(); x++){
				if(Double.isNaN(seq.groundTruthAt(x, 0)))
					continue;
				int maxGT = -1;
				double maxGTValue = Double.NEGATIVE_INFINITY;
				int maxRes = -1;
				double maxResValue = Double.NEGATIVE_INFINITY;
				n++;
				
				for(int y = 0; y < seq.outputSize(); y++){
					if(maxGTValue < seq.groundTruthAt(x, y)){
						maxGT = y;
						maxGTValue = seq.groundTruthAt(x, y);
					}
				}
				
				for(int y = 0; y < seq.outputSize(); y++){
					if(maxResValue < seq.resultAt(x, y)){
						maxRes = y;
						maxResValue = seq.resultAt(x, y);
					}
				}
				
				if(maxGT == maxRes) ok++;
			}
		}
		ok /= n;
		return ok;
	}


	@Override
	public void printResults(String fileName) {
		// TODO Auto-generated method stub

	}

	@Override
	public void printResultsAndGnuplot(String fileName) {
		// TODO Auto-generated method stub

	}

}

package ch.eonum.pipeline.classification.meta;


import ch.eonum.pipeline.classification.Classifier;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;

/**
 * AdaBoost for Regression.
 * @author tim
 *
 */
public class AdaBoostRegression<E extends Instance> extends AdaBoost<E> {
	public AdaBoostRegression(Classifier<E> baseClassifier, Features features) {
		super(baseClassifier, features);
	}

	@Override
	protected double getWeight(double maxOutcome, Instance each) {
		return Math.pow((each.outcome - each.get("result"))/maxOutcome, 2) > 0.2 ? 1.0 : 0.0;
	}
	
	@Override
	public DataSet<E> test() {
		for(E e : testDataSet){
			e.putResult("result", 0.);
			e.putResult("resultTemp", 0.);
		}
		double weightSum = 0.0;
		int M = (int) this.getDoubleParameter("m");
		for(int iteration = 0; iteration < M; iteration++){
			weightSum += modelWeights[iteration];
			baseClassifier.setTestSet(testDataSet);
			baseClassifier.setBaseDir(baseDir + iteration + "/");
			testDataSet = baseClassifier.test();
			for(Instance each : testDataSet)
				each.putResult("resultTemp", each.getResult("resultTemp") + each.getResult("result") * modelWeights[iteration]);
				
		}
		for(Instance each : testDataSet)
				each.putResult("result", each.getResult("resultTemp") / weightSum);
		
		return this.testDataSet;
	}
}

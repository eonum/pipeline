package ch.eonum.pipeline.classification;

import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Log;

/**
 * Wraps a standard classifier to produce an outcome which is transformed using
 * linear regression on the outcome. Hence classifiers can be used for
 * regression tasks when setting the outcome to the probability of one class
 * (Until now this did never turn out to be very accurate!) or the outcome of a
 * regressor can be further normalized/optimized using linear regression (Can be
 * useful to reduce bias of an underlying regressor).
 * 
 * @author tim
 * 
 */
public class LinearRegressionWrapper<E extends Instance> extends Classifier<E> {

	/** underlying inner classifier/regressor. */
	private Classifier<E> innerClassifier;
	/** coefficient b in regression equation. (y = bx + a). */
	private double b;
	/** coefficient a in regression equation. (y = bx + a). */
	private double a;

	public LinearRegressionWrapper(Classifier<E> innerClassifier) {
		this.innerClassifier = innerClassifier;
		this.testDataSet = innerClassifier.testDataSet;
		this.trainingDataSet = innerClassifier.trainingDataSet;
	}

	@Override
	public void train() {
		this.innerClassifier.setTrainingSet(trainingDataSet);
		this.innerClassifier.setTestSet(testDataSet);
		this.innerClassifier.train();
		// x is the predicted result, y is the real outcome
		this.innerClassifier.setTestSet(trainingDataSet);
		this.trainingDataSet = this.innerClassifier.test();
		this.innerClassifier.setTestSet(testDataSet);
		double sumX = 0.0;
		double sumXSquare = 0.0;
		double sumY = 0.0;
		double sumXY = 0.0;
		int n = 0;
		for(Instance each : this.trainingDataSet){
			double x = each.get("result");
			double y = each.outcome;
			sumX += x;
			sumY += y;
			sumXSquare += x*x;
			sumXY += x*y;
			n++;
		}
		this.b = (n*sumXY - sumX*sumY)/(n*sumXSquare - Math.pow(sumX, 2));
		this.a = (sumY/n) - this.b*(sumX/n);
		Log.puts("Regression function: " + this.b + "x +" + this.a);
	}
	
	@Override
	public DataSet<E> test(){
		this.innerClassifier.setTestSet(testDataSet);
		DataSet<E> data = this.innerClassifier.test();
		for(Instance each : data){
			each.putResult("result", b * each.get("result") + a);
			if(each.getResult("result") < 0.0)
				each.putResult("result", 0.0);
		}
		return data;
	}
	
	@Override
	public void setFeatures(Features features){
		super.setFeatures(features);
		this.innerClassifier.setFeatures(features);
	}

}

package ch.eonum.pipeline.classification;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;


import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import ch.eonum.pipeline.classification.libsvm.svm_train;
import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Log; 


/**
 * SVM wrapper for the libsvm implementation in package {@link ch.eonum.pipeline.classification.libsvm.SVM}.
 * @author tim
 *
 */
public class SupportVectorMachine<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("c", "cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)");
		PARAMETERS.put("kernelType","kernel type. \n"
								+ "0 -- linear: u'*v \n"
								+ "1 -- polynomial: (gamma*u'*v + coef0)^degree \n"
								+ "2 -- radial basis function: exp(-gamma*|u-v|^2) (default)\n"
								+ "3 -- sigmoid: tanh(gamma*u'*v + coef0) \n"
								+ "4 -- precomputed kernel (kernel values in training_set_file) \n");
		PARAMETERS.put("g", "gamma : set gamma in kernel function (default 1/num_features)");
		PARAMETERS.put("w", "weight : set the parameter C of class i to weight*C, for C-SVC (default 1)");
		PARAMETERS.put("m", "cachesize : set cache memory size in MB (default 100)");
		PARAMETERS.put("scale", "do sclaing or not");
	}

	protected ArrayList<String> lineNumberIdMap;
	
	/** do scaling before training and testing. default: false */
	private boolean scale;

	private Vector<Double> vy;
	private Vector<svm_node[]> vx;

	private Map<Integer, Double> rangesUpper;
	private Map<Integer, Double> rangesLower;

	/** whether to train a SVC or SVR model for probability estimates. */
	private boolean probabilityEstimates;

	private boolean writeClassProbabilities;

	private svm_model model;

	private boolean recentlyTrained;
	
	public SupportVectorMachine(){
		super();
		this.init();
	}

	public SupportVectorMachine(Features features) {
		super();
		this.init();
		this.setFeatures(features);
	}
	
	private void init() {
		classify = true;
		this.setSupportedParameters(SupportVectorMachine.PARAMETERS);
		this.putParameter("kernelType", "2");
		this.putParameter("c", 1.0);
		this.putParameter("g", -1.0);
		this.putParameter("w", -1.0);
		this.putParameter("m", 100);
		this.scale = false;
		this.probabilityEstimates = true;
	}

	@Override
	public void train() {
		this.prepareClasses();
		
		this.readToSVMLightFormat(this.trainingDataSet);
		if(scale){
			this.createScale();
			this.scaleInputs();
		}
		
		svm_parameter params = this.createParameters();
		svm_problem prob = this.createSVMProblem(params);
		
		svm_train trainer = new svm_train(params, prob, baseDir + "svm.model");
		trainer.run();
		recentlyTrained = true;
	}
	
	@Override
	public DataSet<E> test() {
		this.readToSVMLightFormat(this.testDataSet);
		if(scale)
			this.scaleInputs();

		if(model == null || recentlyTrained){
			try {
				model = svm.svm_load_model(baseDir + "svm.model");
				recentlyTrained = false;
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		if (probabilityEstimates) {
			if (svm.svm_check_probability_model(model) == 0) {
				Log.puts("Model does not support probabiliy estimates\n");
				System.exit(1);
			}
		} else if (svm.svm_check_probability_model(model) != 0)
			Log.puts("Model supports probability estimates, but disabled in prediction.\n");
		
		int svm_type = svm.svm_get_svm_type(model);
		int nr_class = svm.svm_get_nr_class(model);
		double[] prob_estimates = null;

		if (probabilityEstimates) {
			if (svm_type == svm_parameter.EPSILON_SVR
					|| svm_type == svm_parameter.NU_SVR) {
				System.out
						.print("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma="
								+ svm.svm_get_svr_probability(model) + "\n");
			} else {
				int[] labels = new int[nr_class];
				svm.svm_get_labels(model, labels);
				prob_estimates = new double[nr_class];
			}
		}
		for (int i = 0; i < testDataSet.size(); i++) {
			svm_node[] x = this.vx.get(i);
			Instance inst = this.testDataSet.get(i);

			double v;
			if (probabilityEstimates
					&& (svm_type == svm_parameter.C_SVC || svm_type == svm_parameter.NU_SVC)) {
				v = svm.svm_predict_probability(model, x, prob_estimates);
				
				double maxValue = Double.NEGATIVE_INFINITY;
				for (int j = 0; j < nr_class; j++)
					if (prob_estimates[j] > maxValue) {
						maxValue = prob_estimates[j];
						if(this.writeClassProbabilities)
							inst.putResult("result" + classes.getFeatureByIndex(model.label[j]),
									prob_estimates[j]);
					}
				inst.putResult("result", maxValue);
				inst.label = classes.getFeatureByIndex((int)v);
			} else {
				v = svm.svm_predict(model, x);
				inst.label = classes.getFeatureByIndex((int) v);
			}
		}
		
		return testDataSet;
	}
	
	private svm_parameter createParameters() {
		svm_parameter params = new svm_parameter();
		params.cache_size = this.getIntParameter("m");
		params.kernel_type = Integer.parseInt(this.getStringParameter("kernelType"));
		params.probability = probabilityEstimates ? 1 : 0;
		params.C = this.getDoubleParameter("c");
		if(this.getDoubleParameter("g") != -1.0)
			params.gamma = this.getDoubleParameter("g");
		return params;
	}

	private svm_problem createSVMProblem(svm_parameter params) {
		svm_problem prob = new svm_problem();
		prob.l = vy.size();
		prob.x = new svm_node[prob.l][];
		for (int i = 0; i < prob.l; i++)
			prob.x[i] = vx.elementAt(i);
		prob.y = new double[prob.l];
		for (int i = 0; i < prob.l; i++)
			prob.y[i] = vy.elementAt(i);

		int max_index = features.size() - 1;
		if (params.gamma == 0 && max_index > 0)
			params.gamma = 1.0 / max_index;

		if (params.kernel_type == svm_parameter.PRECOMPUTED)
			for (int i = 0; i < prob.l; i++) {
				if (prob.x[i][0].index != 0) {
					System.err
							.print("Wrong kernel matrix: first column must be 0:sample_serial_number\n");
					System.exit(1);
				}
				if ((int) prob.x[i][0].value <= 0
						|| (int) prob.x[i][0].value > max_index) {
					System.err
							.print("Wrong input format: sample_serial_number out of range\n");
					System.exit(1);
				}
			}
		return prob;
	}

	private void scaleInputs() {
		for(svm_node[] x : vx)
			for(svm_node xi : x){
				double min = rangesLower.get(xi.index);
				double max = rangesUpper.get(xi.index);
				if (max == min)
					xi.value = 0.;
				else if (xi.value == min)
					xi.value = 0;
				else if (xi.value == max)
					xi.value = 1.0;
				else 
					xi.value = (xi.value - min) / (max - min);
			}
		
	}

	private void createScale() {
		this.rangesUpper = new HashMap<Integer, Double>();
		this.rangesLower = new HashMap<Integer, Double>();
		for(int i = 0; i < features.size(); i++){
			rangesUpper.put(i, Double.NEGATIVE_INFINITY);
			rangesLower.put(i, Double.POSITIVE_INFINITY);
		}
		for(svm_node[] x : vx){
			int previousIndex = 0;
			for(svm_node xi : x){
				for (int i = previousIndex + 1; i< xi.index; i++)
				{
					rangesUpper.put(i, Math.max(rangesUpper.get(i), 0));
					rangesLower.put(i, Math.min(rangesLower.get(i), 0));
				}
				previousIndex = xi.index;
				if(xi.value > rangesUpper.get(xi.index))
					rangesUpper.put(xi.index, xi.value);
				if(xi.value < rangesLower.get(xi.index))
					rangesLower.put(xi.index, xi.value);
			}
		}
	}

	private void readToSVMLightFormat(DataSet<E> data) {
		this.vy = new Vector<Double>();
		this.vx = new Vector<svm_node[]>();
		int max_index = 0;

		for(Instance each : data) {
			Integer classIndex = classes.getIndexFromFeature(each.groundTruth);
			vy.addElement((double) (classIndex == null ? 0 : classIndex)); // #TODO take outcome for regression
			
			int m = each.features().size();
			for(String f : each.features())
				if(!features.hasFeature(f))
					m--;
			
			svm_node[] x = new svm_node[m];
			int j = 0;
			for (int i = 0; i < features.size(); i++) {
				String feature = features.getFeatureByIndex(i);
				if(each.hasFeature(feature)){
					x[j] = new svm_node();
					x[j].index = i;
					x[j].value = each.get(feature);
					j++;
				}
			}
			if (m > 0)
				max_index = Math.max(max_index, x[m - 1].index);
			vx.addElement(x);
		}
	}
	
	@Override
	public void loadSerializedState(File file) throws IOException {
		super.loadSerializedState(file);
		this.scale = this.getBooleanParameter("scale");
	}

	public void enableScaling() {
		this.putParameter("scale", true);
		this.scale = true;
	}
	
	public void disableScaling(){
		this.putParameter("scale", false);
		this.scale = false;
	}

	public boolean isScaling() {
		return scale;
	}
	
	public void setProbabilityEstimates(boolean pe){
		this.probabilityEstimates = pe;
	}

	public void writeClassProbabilities() {
		this.writeClassProbabilities = true;
	}

}

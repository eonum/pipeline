package ch.eonum.pipeline.classification;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


import ch.eonum.pipeline.core.DataSet;
import ch.eonum.pipeline.core.Features;
import ch.eonum.pipeline.core.Instance;
import ch.eonum.pipeline.util.Shell;


/**
 * Support Vector Machine for Regression using libsvm.
 * SVM wrapper for the libsvm implementation in package {@link ch.eonum.pipeline.classification.libsvm.SVM}.
 * 
 * @author tim
 *
 */
public class SupportVectorRegression<E extends Instance> extends Classifier<E> {
	protected static final Map<String, String> PARAMETERS = new HashMap<String, String>();
	
	static {
		PARAMETERS.put("c", "cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)");
		PARAMETERS.put("kernelType","kernel type. \n"
								+ "0 -- linear: u'*v \n"
								+ "1 -- polynomial: (gamma*u'*v + coef0)^degree \n"
								+ "2 -- radial basis function: exp(-gamma*|u-v|^2) (default)\n"
								+ "3 -- sigmoid: tanh(gamma*u'*v + coef0) \n"
								+ "4 -- precomputed kernel (kernel values in training_set_file) \n");
		PARAMETERS.put("svmType","svm type. \n"
				+ "3 -- epsilon-SVR (default) \n"
				+ "4 -- nu-SVR \n");

		PARAMETERS.put("g", "gamma : set gamma in kernel function (default 1/num_features)");	
		PARAMETERS.put("m", "cachesize : set cache memory size in MB (default 100)");
		PARAMETERS.put("n", "nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)");
		PARAMETERS.put("p", "epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)");
	}

	private ArrayList<String> lineNumberIdMap;
	
	/** do scaling before training and testing. default: false */
	private boolean scale;
	
	public SupportVectorRegression(){
		super();
		this.init();
	}

	public SupportVectorRegression(Features features){
		super();
		this.init();
		this.setFeatures(features);
	}
	
	private void init() {
		this.setSupportedParameters(SupportVectorRegression.PARAMETERS);
		this.putParameter("kernelType", "2");
		this.putParameter("svmType", "3");
		this.putParameter("c", 1.0);
		this.putParameter("g", -1.0);
		this.putParameter("m", 100);
		this.putParameter("n", 0.5);
		this.putParameter("p", 0.1);
		this.scale = false;
	}

	@Override
	public void train() {
		String trainFileName = this.getBaseDir() + "training.txt";
		printSVMInputFile(this.trainingDataSet, trainFileName);
		if(scale) {
			Shell.executeCommand("svm-scale", "svm-scale -l 0 -u 1 -s " + baseDir + "range " + trainFileName, trainFileName + ".scale");
			Shell.executeCommand("cp", "cp " + trainFileName + ".scale " + trainFileName);
		}
		
		String additionalParameters = "";
		if(this.getDoubleParameter("g") != -1.0)
			additionalParameters += "-g " + getDoubleParameter("g") + " ";
		if(this.getDoubleParameter("n") != 0.5)
			additionalParameters += "-n " + getDoubleParameter("n") + " ";
		if(this.getDoubleParameter("p") != 0.1)
			additionalParameters += "-p " + getDoubleParameter("p") + " ";
		if(this.getDoubleParameter("c") != 1.0)
			additionalParameters += "-c " + getDoubleParameter("c") + " ";
		
		Shell.executeCommand("svm-train", "svm-train -m "
				+ getIntParameter("m") + " -s "
				+ getStringParameter("svmType")
				+ " -t "
				+ getStringParameter("kernelType")
				+ " " + additionalParameters
				+ trainFileName + " " + baseDir + "svm.model");
	}

	@Override
	public DataSet<E> test() {
		String testFileName = this.getBaseDir() + "test.txt";
		printSVMInputFile(this.testDataSet, testFileName);
		if(scale) {
			Shell.executeCommand("svm-scale", "svm-scale -r " + baseDir + "range " + testFileName, testFileName + ".scale");
			Shell.executeCommand("cp", "cp " + testFileName + ".scale " + testFileName);
		}
		
		Shell.executeCommand("svm-predict",
				"svm-predict "
						+ testFileName + " " + this.baseDir + "svm.model "
						+ this.baseDir + "svm-out-test.txt");
		
//		try {
//			svm_predict.main(new String[]{testFileName, this.baseDir + "svm.model", this.baseDir + "svm-out-test.txt"});
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		this.readSVMOutputFile(this.testDataSet, this.baseDir + "svm-out-test.txt");
		return testDataSet;
	}
	
	/**
	 * write a data set in the libSVM format to a file.
	 * @param data
	 * @param file
	 */
	public void printSVMInputFile(DataSet<E> data, String file){
		try {
			this.lineNumberIdMap = new ArrayList<String>();
			PrintStream p = new PrintStream(new FileOutputStream(file));
			int lineNumber = 0;
			for (Instance inst : data) {
				String line = inst.outcome + " ";
				for(int i = 0; i < features.size(); i++){
					String feature = features.getFeatureByIndex(i);
					if(inst.get(feature) != 0)
						line += (i+1) + ":" + inst.get(feature) + " ";
				}
				if(line.contains("null"))
					continue;
				p.println(line);
				this.lineNumberIdMap.add(lineNumber++, inst.id);
			}
			p.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * read an output file from libSVM.
	 * @param data
	 * @param file
	 */
	public void readSVMOutputFile(DataSet<E> data, String file){
		Map<String, Double> results = new HashMap<String, Double>();

		try {
			FileInputStream fstream = new FileInputStream(file);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;
			// set all likelihoods to 1
			for (Instance inst : data)
				inst.putResult("result", 0.0f);
			int lineNumber = 0;
			
			while ((strLine = br.readLine()) != null) {
				String[] split = strLine.split(" ");
				double prediction = Double.valueOf(split[0]);
				results.put(this.lineNumberIdMap.get(lineNumber), prediction);
				lineNumber++;
			}
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for (Instance inst : data) {
			inst.putResult("result", results.get(inst.id));
		}
	}

	public void enableScaling() {
		this.scale = true;
	}
	
	public void disableScaling(){
		this.scale = false;
	}

	public boolean isScaling() {
		return scale;
	}

}

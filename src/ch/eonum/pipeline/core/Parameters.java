package ch.eonum.pipeline.core;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import ch.eonum.pipeline.util.Log;
import ch.eonum.pipeline.util.json.JSON;

/**
 * Parameter object providing meta parameters. Every object which has meta
 * parameters, can inherit Parameters. A meta parameter is a parameter which
 * is/can be validated during validation.
 * 
 * @author tp
 * 
 */
public class Parameters {

	private Map<String, Double> doubleParameters;
	private Map<String, Integer> intParameters;
	private Map<String, String> stringParameters;
	private Map<String, Boolean> booleanParameters;
	
	/**
	 * a list with all supported parameters and their description (should also
	 * contain the default value)
	 */
	private Map<String, String> supportedParameters;
	
	private void checkParam(String param) {
		if(!supportedParameters.containsKey(param))
			Log.error("The parameter " + param + " is not supported");
	}

	public void putParameter(String param, double value) {
		checkParam(param);
		this.doubleParameters.put(param, value);
	}
	
	public void putParameter(String param, int value) {
		checkParam(param);
		this.intParameters.put(param, value);
	}
	
	public void putParameter(String param, String value) {
		checkParam(param);
		this.stringParameters.put(param, value);
	}
	
	public void putParameter(String param, boolean value) {
		checkParam(param);
		this.booleanParameters.put(param, value);
	}

	public synchronized double getDoubleParameter(String param) {
		return this.doubleParameters.get(param);
	}
	
	public synchronized int getIntParameter(String param) {
		return this.intParameters.get(param);
	}
	
	public synchronized String getStringParameter(String param) {
		return this.stringParameters.get(param);
	}
	
	public synchronized boolean getBooleanParameter(String param) {
		return this.booleanParameters.get(param);
	}
	
	public Parameters() {
		this.doubleParameters = new HashMap<String, Double>();
		this.intParameters = new HashMap<String, Integer>();
		this.stringParameters = new HashMap<String, String>();
		this.booleanParameters = new HashMap<String, Boolean>();
		this.setSupportedParameters(new HashMap<String, String>());
	}
	
	/**
	 * print all parameters into a file.
	 * 
	 * @param file
	 * @throws IOException 
	 */
	public void printParameters(File file) throws IOException {
		FileWriter fstream = new FileWriter(file);
		BufferedWriter out = new BufferedWriter(fstream);
		out.write(this.getParametersAsJSONString());
		out.close();
	}
	
	protected String getParametersAsJSONString() {
		Map<String, Object> bdbo = this.asMap();
		return bdbo.toString();
	}

	public Map<String, Object> asMap() {
		Map<String, Object> dbo = new HashMap<String, Object>();
		dbo.put("stringParams", this.stringParameters);
		dbo.put("doubleParams", this.doubleParameters);
		dbo.put("intParams", this.intParameters);
		dbo.put("booleanParams", this.booleanParameters);
		return dbo;
	}

	public void printParameters(String fileName) throws IOException{
		this.printParameters(new File(fileName));
	}
	
	/**
	 * load parameters from file.
	 * @param file
	 * @throws IOException 
	 */
	public void load(File file) throws IOException {
		Map<String, Object> data = JSON.readJSON(file);
		this.load(data);
	}
	
	@SuppressWarnings("unchecked")
	protected void load(Map<String, Object> data) {
		this.booleanParameters = (Map<String, Boolean>) data.get("booleanParams");
		this.stringParameters = (Map<String, String>) data.get("stringParams");
		this.doubleParameters = (Map<String, Double>) data.get("doubleParams");
		this.intParameters = (Map<String, Integer>) data.get("intParams");
	}

	/**
	 * load parameters from file.
	 * @param fileName
	 * @throws IOException 
	 */
	public void load(String fileName) throws IOException {
		this.load(new File(fileName));
	}

	public void setSupportedParameters(Map<String, String> supportedParameters) {
		this.supportedParameters = supportedParameters;
	}

	public Map<String, String> getSupportedParameters() {
		return supportedParameters;
	}
}

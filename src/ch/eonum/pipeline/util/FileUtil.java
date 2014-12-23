package ch.eonum.pipeline.util;

import java.io.File;

public class FileUtil {

	/**
	 * create a folder if it does not already exist.
	 * @param string
	 */
	public static void mkdir(String folderName) {
		File folder = new File(folderName);
		if(!folder.exists())
			folder.mkdir();
		if(!folder.isDirectory()){
			Log.error(folderName + " is supposed to be a folder, but it is not.");
			System.exit(-11);
		}
	}

}

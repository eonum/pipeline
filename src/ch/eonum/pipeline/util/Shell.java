package ch.eonum.pipeline.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;

/**
 * Access the system shell and redirect standard and error outputs.
 * 
 * @author tim
 * 
 */
public class Shell {
	/**
	 * Execute an external command. Print out standard error and standard out
	 * and wait for the process to exit.
	 * 
	 * @param programName
	 * @param command
	 */
	public static void executeCommand(String programName, String command) {
		executeCommand(programName, command, null);
	}

	/**
	 * Execute an external command. Print out standard error and standard out
	 * and wait for the process to exit. write the standard output into a file
	 * specified by stdoutFile.
	 * 
	 * @param programName
	 * @param command
	 * @param stdoutFile
	 *            for standard out
	 */
	public static void executeCommand(String programName, String command,
			String stdoutFile) {
		try {
            
		    Process p = Runtime.getRuntime().exec(command);
			InputStream stderr = p.getErrorStream();
			InputStream stdout = p.getInputStream();

			String line;
			// clean up if any output in stdout
			BufferedReader brCleanUp = new BufferedReader(
					new InputStreamReader(stdout));

			if (stdoutFile != null) {
				PrintStream ps = new PrintStream(stdoutFile);
				while ((line = brCleanUp.readLine()) != null) {
					ps.println(line);
				}
				ps.close();
			} else
				while ((line = brCleanUp.readLine()) != null) {
					Log.puts("[" + programName + " out] " + line);
				}
			brCleanUp.close();

			// clean up if any output in stderr
			brCleanUp = new BufferedReader(new InputStreamReader(stderr));
			while ((line = brCleanUp.readLine()) != null) {
				Log.warn("[" + programName + " error] " + line);
			}
			brCleanUp.close();

			p.waitFor();

		} catch (IOException e) {
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
}

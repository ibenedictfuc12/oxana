const { spawn } = require('child_process');

function generateImage() {
  return new Promise((resolve, reject) => {
    const py = spawn('python', ['examples/generate_image.py']);

    let output = '';
    let errorOutput = '';

    py.stdout.on('data', data => {
      output += data.toString();
    });

    py.stderr.on('data', data => {
      errorOutput += data.toString();
    });

    py.on('close', code => {
      if (code !== 0) {
        reject(`Process exited with code ${code}\nStderr: ${errorOutput}`);
      } else {
        resolve(output);
      }
    });
  });
}

(async () => {
  try {
    const result = await generateImage();
    console.log(`Image generation result: ${result}`);
  } catch (err) {
    console.error(err);
  }
})();
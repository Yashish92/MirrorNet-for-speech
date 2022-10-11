## Learning Articulatory Representations with the MIRRORNET

This website presents audio reconstructions and auditory spectrograms for experiments done with MirrorNet on speech production

### Brief Intro

Experiments to understand the sensorimotor neural interactions in the human cortical speech system support the existence of a bidirectional flow of interactions between the auditory and motor regions. Their key function is to enable the brain to ’learn’ how to control the vocal tract for speech production. This idea is the impetus for the recently proposed ”MirrorNet”, a constrained autoencoder architecture. 

### 1. Articulatory-to-acoustic synthesizer

![Model architecture](model_figures/tv_synthesizer.png)

#### -- Auditory spectrograms and audio samples

Tables below show output auditory spectrograms and their corresponding acoustic output obtained by inverting the auditory spectrograms. The two input speech utterenaces are from a male speaker and a feamle speaker in the test split. 

|  Description   |  Audio   |  Auditory spectrogram   |
| --- | --- | --- |
|  Input Audio (Female)   |  <audio controls="controls"><source src="TV_syn_audio/original/RealSpectrogram4.mp3"></audio>  |  <img width="300px" height="160px" src="TV_syn_spectrograms/original/sample4_orig.png">   |
|  Fully trained with source features   |  <audio controls="controls"><source src="TV_syn_audio/fully_trained/decoderSpectrogram4.mp3"></audio>   |  <img width="300px" height="160px" src="TV_syn_spectrograms/fully_trained/sample4_fully.png">   |
|  Fully trained 'without' source features  |  <audio controls="controls"><source src="TV_syn_audio/6TV_syn/decoderSpectrogram4_6tvs.mp3"></audio>   |  <img width="300px" height="160px" src="TV_syn_spectrograms/6TV_syn/sample4_6TV.png">   |
|  Lightly trained with source features   |  <audio controls="controls"><source src="TV_syn_audio/lightly_trained/decoderSpectrogram4_dev.mp3"></audio>   |  <img width="300px" height="160px" src="TV_syn_spectrograms/lightly_trained/sample4_light.png"> |

|  Description   |  Audio   |  Auditory spectrogram   |
| --- | --- | --- |
|  Input Audio (Female)   |  <audio controls="controls"><source src="TV_syn_audio/original/RealSpectrogram6.mp3"></audio>  |  <img width="300px" height="160px" src="TV_syn_spectrograms/original/sample6_orig.png">   |
|  Fully trained with source features   |  <audio controls="controls"><source src="TV_syn_audio/fully_trained/decoderSpectrogram6.mp3"></audio>   |  <img width="300px" height="160px" src="TV_syn_spectrograms/fully_trained/sample6_fully.png">   |
|  Fully trained 'without' source features  |  <audio controls="controls"><source src="TV_syn_audio/6TV_syn/decoderSpectrogram6_6tvs.mp3"></audio>   |  <img width="300px" height="160px" src="TV_syn_spectrograms/6TV_syn/sample6_6TV.png">   |
|  Lightly trained with source features   |  <audio controls="controls"><source src="TV_syn_audio/lightly_trained/decoderSpectrogram6.mp3"></audio>   |  <img width="300px" height="160px" src="TV_syn_spectrograms/lightly_trained/sample6_light.png">   |

### 2. MirrorNet with the articulatory synthesizer

![Model architecture](model_figures/model_archi.png)

#### -- Auditory spectrograms and audio samples

Tables below show output auditory spectrograms and their corresponding acoustic output obtained by inverting the auditory spectrograms. Here the articulatory synthesizers developed in the previous step are used as the vocal tract model in the MirrorNet. 

<div class="figure">
    <table class="audioTable">
        <tr>
<!--             <th rowspan="2">Audio</th> -->
            <th>Description</th>
            <th>Audio</th>
	    	<th>Auditory spectrogram</th>
<!--             <th rowspan="2">Audio</th> -->
        </tr>
        <tr>
			<td> Input Audio 
            <td><audio controls="controls"><source src="MirrorNet_audio/original/wave2_orig.mp3"></audio></td>
<!--             <td>PARAMS IMG</td> -->
            <td><img width="300px" height="160px" src="MirrorNet_specs/sample2_orig.png"></td>
        </tr>
        <tr>
			<td> With Initialization and Fully trained synthesizer 
            <td><audio controls="controls"><source src="MirrorNet_audio/fully_trained_init/VOCSpectrogram2_init.mp3"></audio></td>
<!--             <td>PARAMS IMG</td> -->
            <td><img width="300px" height="160px" src="MirrorNet_specs/fully_trained_init/sample2_fully.png"></td>
        </tr>
		        <tr>
			<td> 'No' Initialization and Fully trained synthesizer 
            <td><audio controls="controls"><source src="MirrorNet_audio/fully_trained_noinit/VOCSpectrogram2_noinit.mp3"></audio></td>
<!--             <td>PARAMS IMG</td> -->
            <td><img width="300px" height="160px" src="MirrorNet_specs/fully_trained_noinit/sample2_noinit.png"></td>
        </tr>
		        <tr>
			<td> With Initialization and lightly trained synthesizer 
            <td><audio controls="controls"><source src="MirrorNet_audio/light_trained_init/VOCSpectrogram2_light.mp3"></audio></td>
<!--             <td>PARAMS IMG</td> -->
            <td><img width="300px" height="160px" src="MirrorNet_specs/light_trained_init/sample2_light.png"></td>
        </tr>
    </table>
</div>

<div class="figure">
    <table class="audioTable">
        <tr>
<!--             <th rowspan="2">Audio</th> -->
            <th>Description</th>
            <th>Audio</th>
	    	<th>Auditory spectrogram</th>
<!--             <th rowspan="2">Audio</th> -->
        </tr>
        <tr>
			<td> Input Audio 
            <td><audio controls="controls"><source src="MirrorNet_audio/original/wave8_orig.mp3"></audio></td>
<!--             <td>PARAMS IMG</td> -->
            <td><img width="300px" height="160px" src="MirrorNet_specs/sample8_orig.png"></td>
        </tr>
        <tr>
			<td> With Initialization and Fully trained synthesizer 
            <td><audio controls="controls"><source src="MirrorNet_audio/fully_trained_init/VOCSpectrogram8_init.mp3"></audio></td>
<!--             <td>PARAMS IMG</td> -->
            <td><img width="300px" height="160px" src="MirrorNet_specs/fully_trained_init/sample8_fully.png"></td>
        </tr>
		        <tr>
			<td> 'No' Initialization and Fully trained synthesize 
            <td><audio controls="controls"><source src="MirrorNet_audio/fully_trained_noinit/VOCSpectrogram8_noinit.mp3"></audio></td>
<!--             <td>PARAMS IMG</td> -->
            <td><img width="300px" height="160px" src="MirrorNet_specs/fully_trained_noinit/sample8_noinit.png"></td>
        </tr>
		        <tr>
			<td> With Initialization and lightly trained synthesizer 
            <td><audio controls="controls"><source src="MirrorNet_audio/light_trained_init/VOCSpectrogram8_light.mp3"></audio></td>
<!--             <td>PARAMS IMG</td> -->
            <td><img width="300px" height="160px" src="MirrorNet_specs/light_trained_init/sample8_light.png"></td>
        </tr>
    </table>
</div>

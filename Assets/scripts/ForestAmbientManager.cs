using UnityEngine;
using System.Collections;

public class ForestAmbientManager : MonoBehaviour
{
    [Header("Random Forest Sounds")]
    public AudioClip[] ambientSounds;

    public float minDelay = 20f;
    public float maxDelay = 60f;

    private void Start()
    {
        StartCoroutine(RandomAmbientLoop());
    }

    IEnumerator RandomAmbientLoop()
    {
        while (true)
        {
            yield return new WaitForSeconds(Random.Range(minDelay, maxDelay));

            if (AudioManager.Instance == null)
                continue;

            if (ambientSounds.Length == 0)
                continue;

            int index = Random.Range(0, ambientSounds.Length);
            AudioManager.Instance.PlayRandomAmbient(ambientSounds[index]);
        }
    }
}
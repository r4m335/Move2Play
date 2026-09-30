using UnityEngine;
using System.Collections;

public class EnemyAlertPopup : MonoBehaviour
{
    public GameObject alertObject;

    public float showDuration = 30f;   // Increased to 30 seconds

    private Coroutine currentRoutine;

    void Start()
    {
        if (alertObject != null)
        {
            alertObject.SetActive(false);
        }
    }

    public void ShowAlert()
    {
        if (currentRoutine != null)
        {
            StopCoroutine(currentRoutine);
        }

        currentRoutine = StartCoroutine(AlertRoutine());
    }

    IEnumerator AlertRoutine()
    {
        alertObject.SetActive(true);

        yield return new WaitForSeconds(showDuration);

        alertObject.SetActive(false);
    }
}
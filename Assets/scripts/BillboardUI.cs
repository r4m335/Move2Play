using UnityEngine;

public class BillboardUI : MonoBehaviour
{
    void LateUpdate()
    {
        if (Camera.main == null) return;

        transform.forward =
            Camera.main.transform.forward;
    }
}
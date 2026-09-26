import unittest


class EyeSupportTests(unittest.TestCase):
    def test_endpoints_do_not_imply_sustained_support(self):
        from training.visual.eye_support import summarize_support, require_eye_support
        rows=[dict(subject_id='7',label_known=True,target=1,subtype='closed_eye',eye_endpoint_available=True,eye_duration_eligible=False),
              dict(subject_id='14',label_known=True,target=1,subtype='closed_eye',eye_endpoint_available=False,eye_duration_eligible=False),
              dict(subject_id='14',label_known=False,target=0,subtype='unknown',eye_endpoint_available=True,eye_duration_eligible=True),
              dict(subject_id='14',label_known=True,target=0,subtype='none',eye_endpoint_available=True,eye_duration_eligible=True)]
        report=summarize_support(rows)
        self.assertEqual(report['by_subtype']['closed_eye'],dict(positive=2,endpoint_available=1,duration_eligible=0))
        self.assertEqual(report['negative_duration_eligible'],1)
        with self.assertRaisesRegex(ValueError,'continuous|support'):require_eye_support(report)
        rows[0]['eye_duration_eligible']=True
        require_eye_support(summarize_support(rows))


if __name__=='__main__':unittest.main()

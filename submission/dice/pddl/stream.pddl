(define (stream dice-fine-plan-streams)
    (:stream move-stream
        :inputs (?from ?to)
        :domain (and (AtConfig ?from) (AtConfig ?to))
        :outputs (?x)
        :certified (ValidMotion ?from ?t ?to))
    (:stream push-stream
        :inputs (?from ?to)
        :domain (and (AtConfig ?from) (AtConfig ?to))
        :outputs (?x)
        :certified (and ValidPush ?from ?t ?to) )
  )